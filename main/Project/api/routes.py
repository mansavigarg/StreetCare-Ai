import hmac
import secrets
import threading
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from langgraph.types import Command

from access.password_security import hash_password
from access.permissions import authorize_district, authorize_request
from agentic import nodes
from agentic.graph import build_workflow
from api.schemas import (ChatRequest, ChatResponse, ChatTurnResponse, DecisionRequest, DefectTypeFilter,
                         ReportDetailResponse, ReviewCallsResponse, ReviewRecordResponse, ReviewRequest,
                         ReviewStatusValue, SignedInUserResponse, SignedOutResponse, SignInRequest, QueueRow)
from chat.conversation_memory import recall_conversation
from chat.record_answer import answer_report_record, answer_street_figure
from chat.street_care_assistant import answer_from_documents
from config import config
from knowledge.standard_reading import read_defect_standard
from modeling.explainability import build_explainers, explain_report, summarise_queue
from persistence.database import session_factory
from persistence.models import AgentRun, ChatTurn, ReportReview, UserAccount
from persistence.repository import (fetch_chat_turns, fetch_llm_calls, fetch_review_records, fetch_user_account,
                                    insert_chat_turn, update_agent_run, update_session_token)
from tools.decision_tools import find_decision_refusal, find_review_conflict
from tools.report_data_tools import describe_record, load_report_history, rank_report_queue

review_workflow = build_workflow(config.CHECKPOINT_DATABASE_FILE)
report_history = load_report_history()
report_profiles, report_details, report_queue = {}, {}, []
for position, reference in enumerate(rank_report_queue(nodes.open_reports), start=1):
    context = nodes.gather_report_context(reference)
    profile, screening, count = context["report_profile"], context["screening"], context["register_count"]
    described = describe_record(profile)
    report_profiles[reference] = profile
    report_details[reference] = {
        "queue_position": position, "report": described,
        **{key: context[key] for key in ("report_documents", "register_count", "evidence_findings", "disagreements",
                                         "likelihood", "crew_hours")},
        "screening": {**screening, "requirements": [{**item, "rules": {key: float(value) for key, value in
                                                                        item["rules"].items()}}
                                                    for item in screening["requirements"]]}}
    report_queue.append({
        "queue_position": position, "report_reference": reference,
        "revision_number": profile[config.REVISION_NUMBER_COLUMN],
        **{column: described[column] for column in (config.DEFECT_TYPE_COLUMN, config.LINKED_DEFECT_TYPE_COLUMN,
                                                    config.STREET_NAME_COLUMN, "district", "report_channel")},
        "document_count": len(context["report_documents"]),
        **{column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS},
        "live_permit_count": count[config.LIVE_STREET_WORKS_COLUMN],
        "conflicting_permit_count": count[config.CONFLICTING_STREET_WORKS_COLUMN],
        "coordination_trigger_count": len(screening["coordination_triggers"]), "assessment": screening["assessment"],
        **context["likelihood"], "expected_crew_hours": context["crew_hours"]["expected_crew_hours"],
        "overrun_expected": context["crew_hours"]["overrun_expected"],
        "disagreement_count": len(context["disagreements"])})
report_locks = {reference: threading.Lock() for reference in report_profiles}
explanation_cache = {}


def _prepare_explainers():
    standards = {kind: read_defect_standard(kind, nodes.knowledge_documents)
                 for kind in config.DEFECT_STANDARD_DOCUMENTS}
    return build_explainers(nodes.repeat_model, nodes.crew_hours_model, standards)


def _warm_explainers():
    try:
        _prepare_explainers()
    except Exception:  # the endpoints report the problem when they are asked; the service carries on
        pass


threading.Thread(target=_warm_explainers, daemon=True).start()
public_router = APIRouter()
guarded_router = APIRouter(dependencies=[Depends(authorize_request)])
SessionTokenHeader = Annotated[str | None, Header(alias=config.SESSION_TOKEN_HEADER)]
SignedInUser = Annotated[dict, Depends(authorize_request)]


def _report_statuses():
    with session_factory() as session:
        records = fetch_review_records(session, ReportReview.report_reference.in_(list(report_profiles)))
    return {record["report_reference"]: record["status"] for record in records
            if record["revision_number"] == report_profiles[record["report_reference"]][config.REVISION_NUMBER_COLUMN]}


def _profile(report_reference):
    if report_reference not in report_profiles:
        raise HTTPException(404, config.UNKNOWN_REPORT_MESSAGE.format(report_reference=report_reference))
    return report_profiles[report_reference]


def _review(review_id):
    with session_factory() as session:
        records = fetch_review_records(session, ReportReview.review_id == review_id)
    if not records:
        raise HTTPException(404, config.REVIEW_NOT_FOUND_MESSAGE)
    return records[0]


def _checkpoint(record):
    snapshot = review_workflow.get_state({"configurable": {"thread_id": record["thread_id"]}})
    if not snapshot.values:
        raise ValueError(config.CHECKPOINT_MISSING_MESSAGE.format(review_id=record["review_id"],
                                                                  checkpoint_file=config.CHECKPOINT_DATABASE_FILE))
    return snapshot


@public_router.post("/auth/signin", response_model=SignedInUserResponse)
def sign_in_user(sign_in_request: SignInRequest):
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.username == sign_in_request.username)
        if account is None or not hmac.compare_digest(
                hash_password(sign_in_request.password, account["password_salt"]), account["password_hash"]):
            raise HTTPException(401, config.SIGN_IN_FAILED_MESSAGE)
        token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
        update_session_token(session, account["username"], token)
        session.commit()
    return {**account, "session_token": token, "pages": config.ROLE_PAGE_ACCESS[account["role"]],
            "permissions": config.ROLE_PERMISSIONS[account["role"]]}


@public_router.post("/auth/signout", response_model=SignedOutResponse)
def sign_out_user(session_token: SessionTokenHeader = None):
    with session_factory() as session:
        account = session_token and fetch_user_account(session, UserAccount.session_token == session_token)
        if not account:
            raise HTTPException(401, config.SIGN_IN_REQUIRED_MESSAGE)
        update_session_token(session, account["username"], None)
        session.commit()
    return {"username": account["username"], "signed_out": True}


@guarded_router.get("/queue", response_model=list[QueueRow])
def list_report_queue(defect_type: Annotated[DefectTypeFilter, Query()]):
    statuses = _report_statuses()
    return [{**row, "report_status": statuses.get(row["report_reference"], config.REPORT_STATUS_AWAITING_REVIEW)}
            for row in report_queue if defect_type in (config.QUEUE_DEFECT_TYPE_ALL, row[config.DEFECT_TYPE_COLUMN],
                                                       row[config.LINKED_DEFECT_TYPE_COLUMN])]


@guarded_router.get("/queue/{report_reference}", response_model=ReportDetailResponse)
def get_report_detail(report_reference: str):
    _profile(report_reference)
    status = _report_statuses().get(report_reference, config.REPORT_STATUS_AWAITING_REVIEW)
    return {**report_details[report_reference], "report_status": status}


def _explainers_ready():
    try:
        _prepare_explainers()
    except ImportError as error:
        raise HTTPException(503, config.EXPLANATION_UNAVAILABLE_MESSAGE.format(reason=error))


@guarded_router.get("/queue/{report_reference}/explanation")
def get_report_explanation(report_reference: str):
    _profile(report_reference)
    _explainers_ready()
    if report_reference not in explanation_cache:
        explanation_cache[report_reference] = explain_report(report_profiles[report_reference],
                                                             report_details[report_reference]["screening"])
    return {"report_reference": report_reference, **explanation_cache[report_reference]}


@guarded_router.get("/insights")
def get_model_insights():
    _explainers_ready()
    if "queue" not in explanation_cache:
        explanation_cache["queue"] = summarise_queue(
            [(row["report_reference"], report_profiles[row["report_reference"]],
              report_details[row["report_reference"]]["screening"], row) for row in report_queue])
    return explanation_cache["queue"]


@guarded_router.post("/reviews", status_code=201, response_model=ReviewRecordResponse)
def start_report_review(review_request: ReviewRequest, signed_in_user: SignedInUser, http_request: Request):
    reference = review_request.report_reference
    profile = _profile(reference)
    authorize_district(signed_in_user, profile[config.STREET_NAME_COLUMN], reference)
    with report_locks[reference]:
        if reference in _report_statuses():
            raise HTTPException(409, config.REPORT_ALREADY_REVIEWED_MESSAGE.format(
                revision_number=profile[config.REVISION_NUMBER_COLUMN], report_reference=reference))
        thread_id = f"{reference}-{secrets.token_hex(config.IDENTIFIER_HEX_LENGTH // 2).upper()}"
        http_request.state.opened_thread_id = thread_id
        review_workflow.invoke({"report_reference": reference, "coordinator_request": review_request.coordinator_request,
                                "coordinator_id": signed_in_user["username"], "thread_id": thread_id},
                               {"configurable": {"thread_id": thread_id}})
        snapshot = review_workflow.get_state({"configurable": {"thread_id": thread_id}})
    return _review(snapshot.values["review_id"])


@guarded_router.get("/reviews", response_model=list[ReviewRecordResponse])
def list_report_reviews(status_value: Annotated[list[ReviewStatusValue], Query()]):
    with session_factory() as session:
        records = fetch_review_records(session, ReportReview.status.in_(status_value))
    for record in records:
        if record["status"] != config.REVIEW_STATUS_AWAITING:
            continue
        snapshot = _checkpoint(record)
        pending = snapshot.values.get("review_decision")
        if pending is not None:
            record.update(pending_decision=pending["decision"], pending_decided_by=pending["decided_by"],
                         pending_decision_reason=pending["decision_reason"])
    return records


@guarded_router.post("/reviews/{review_id}/decision", response_model=ReviewRecordResponse)
def submit_review_decision(review_id: str, decision_request: DecisionRequest, signed_in_user: SignedInUser):
    record = _review(review_id)
    reference = record["report_reference"]
    if reference not in report_profiles:
        raise ValueError(config.REVIEW_OUTSIDE_QUEUE_MESSAGE.format(review_id=review_id, report_reference=reference))
    authorize_district(signed_in_user, record["street_name"], reference)
    decision = {"decision": decision_request.decision, "decided_by": signed_in_user["username"],
                "decided_by_role": signed_in_user["role"], "decided_by_district": signed_in_user["district"],
                "decision_reason": decision_request.decision_reason}
    with report_locks[reference]:
        record = _review(review_id)
        snapshot = _checkpoint(record)
        state = snapshot.values
        refusal = (find_review_conflict(record["status"], state.get("review_decision"), decision, reference)
                   or find_decision_refusal(decision, state["report_profile"], state["screening"],
                                            state["coordinator_id"]))
        if refusal is not None:
            raise HTTPException(409, refusal)
        waiting = "await_inspector_decision" in snapshot.next
        review_workflow.invoke(Command(resume=decision) if waiting else None,
                               {"configurable": {"thread_id": record["thread_id"]}})
    return _review(review_id)


@guarded_router.get("/reviews/{review_id}/calls", response_model=ReviewCallsResponse)
def get_review_calls(review_id: str):
    record = _review(review_id)
    decision = record["decision"]
    if decision is None:
        pending = _checkpoint(record).values.get("review_decision")
        decision = pending and pending["decision"]
    with session_factory() as session:
        calls = fetch_llm_calls(session, record["run_id"])
    return {"review_id": review_id, "run_id": record["run_id"], "call_count": len(calls),
            "expected_call_count": config.LLM_CALLS_BY_DECISION.get(decision, config.LLM_CALLS_BEFORE_DECISION),
            "total_tokens": sum(call["total_tokens"] for call in calls),
            "total_latency_milliseconds": sum(call["latency_milliseconds"] for call in calls), "calls": calls}


@guarded_router.post("/chat/ask", response_model=ChatResponse)
def ask_street_care_assistant(chat_request: ChatRequest, signed_in_user: SignedInUser):
    username = signed_in_user["username"]
    with session_factory() as session:
        conversation = recall_conversation(session, username, chat_request.question)
    reply = answer_from_documents(nodes.knowledge_collection, nodes.knowledge_documents, chat_request.question,
                                  conversation)
    with session_factory() as session:
        if reply["question_type"] in config.CHAT_REPORT_TYPES:
            reply["answer"] = answer_report_record(session, signed_in_user, nodes.open_reports,
                                                   reply["question_type"], reply["report_reference"])
        elif reply["question_type"] == config.CHAT_TYPE_STREET_FIGURE:
            reply["answer"] = answer_street_figure(report_history, reply["street_name"], reply["defect_type"])
        insert_chat_turn(session, {
            "username": username, "user_role": signed_in_user["role"], "question_text": chat_request.question,
            "answer_text": reply["answer"], "source_documents": ", ".join(reply["source_documents"]),
            **{key: reply[key] for key in ("question_type", "report_reference", "street_name", "defect_type",
                                           "closest_distance", "was_refused")}})
        session.commit()
    return reply


@guarded_router.get("/chat/turns", response_model=list[ChatTurnResponse])
def list_own_chat_turns(signed_in_user: SignedInUser,
                        turn_limit: Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)]):
    with session_factory() as session:
        return fetch_chat_turns(session, ChatTurn.username == signed_in_user["username"], turn_limit)


@guarded_router.get("/audit/chat-turns", response_model=list[ChatTurnResponse])
def list_recent_chat_turns(turn_limit: Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)]):
    with session_factory() as session:
        return fetch_chat_turns(session, ChatTurn.turn_id.is_not(None), turn_limit)


def report_failed_request(http_request, failure):
    thread_id = getattr(http_request.state, "opened_thread_id", None)
    if thread_id is not None:
        with session_factory() as session:
            run = session.query(AgentRun).filter(AgentRun.thread_id == thread_id).first()
            if run is not None:
                update_agent_run(session, run.run_id, config.AGENT_RUN_STATUS_FAILED,
                                 len(fetch_llm_calls(session, run.run_id)))
                session.commit()
    return JSONResponse(status_code=502, content={"detail": str(failure)})


service_application = FastAPI(title=config.API_TITLE)
service_application.include_router(public_router)
service_application.include_router(guarded_router)
for failure_type in (ValueError, genai_errors.APIError, httpx.HTTPError):
    service_application.add_exception_handler(failure_type, report_failed_request)
