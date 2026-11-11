import secrets

from langgraph.types import interrupt

from agentic.agents import (analyse_predictions, analyse_requirements, draft_order_document, draft_resident_update,
                            plan_review, review_evidence, write_review_recommendation)
from agentic.language_model import join_labelled_response
from config import config
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_defect_standard
from knowledge.vector_index import open_vector_index
from modeling.model_registry import load_feature_schema, load_model
from persistence.database import session_factory
from persistence.repository import (insert_agent_run, insert_llm_call, insert_report_review, insert_review_decision,
                                    record_review_outcome, update_agent_run)
from tools.decision_tools import find_decision_refusal
from tools.evidence_tools import find_disagreements, link_evidence_to_requirements, read_document_evidence
from tools.prediction_tools import (describe_expected_crew_hours, estimate_expected_crew_hours,
                                    estimate_repeat_likelihood)
from tools.report_data_tools import (fetch_report, fetch_report_documents, load_open_reports, load_report_documents,
                                     load_street_works_register)
from tools.retrieval_tools import (read_inspection_records, read_policy_section, read_resident_update_guidance,
                                   read_reviewer_checks, read_works_order_guidance)
from tools.review_document_tools import (build_coordination_brief, build_inspection_request_details,
                                         build_works_order_details)
from tools.screening_tools import count_live_street_works, screen_report

open_reports = load_open_reports()
open_report_documents = load_report_documents()
street_works_register = load_street_works_register()
knowledge_documents = extract_text_from_directory(config.DOCS_DIRECTORY)
knowledge_collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
repeat_model, repeat_features = load_model(config.REPEAT_MODEL_FILE), load_feature_schema(config.REPEAT_SCHEMA_FILE)
crew_hours_model = load_model(config.CREW_HOURS_MODEL_FILE)
crew_hours_features = load_feature_schema(config.CREW_HOURS_SCHEMA_FILE)
sentence_model = load_model(config.SENTENCE_MODEL_FILE)


def _identifier(prefix):
    return prefix + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH // 2).upper()


def _record_call(run_id, call_record):
    with session_factory() as session:
        insert_llm_call(session, run_id, call_record)
        session.commit()


def _passages(reader, defect_types):
    return "\n\n".join(config.STANDARD_PASSAGE_TEMPLATE.format(
        defect_type=defect_type, passage=reader(knowledge_collection, defect_type)) for defect_type in defect_types)


def _long_and_different(texts, what):
    short = [name for name, text in texts.items() if len(text.split()) < 8]
    if short or len(set(texts.values())) < len(texts):
        raise ValueError(f"The {what} {', '.join(short) or 'repeat one another'}: each holds at least 8 words of "
                         f"its own")


def gather_report_context(report_reference):
    report = fetch_report(open_reports, report_reference)
    register_count = count_live_street_works(street_works_register, report)
    for column in (config.LIVE_STREET_WORKS_COLUMN, config.CONFLICTING_STREET_WORKS_COLUMN):
        report[column] = register_count[column]
    kinds = [report[config.DEFECT_TYPE_COLUMN], report[config.LINKED_DEFECT_TYPE_COLUMN]]
    standards = {kind: read_defect_standard(kind, knowledge_documents) for kind in kinds
                 if kind in config.DEFECT_STANDARD_DOCUMENTS}
    screening = screen_report(report, standards)
    documents = fetch_report_documents(open_report_documents, report)
    findings = link_evidence_to_requirements(read_document_evidence(sentence_model, documents), screening)
    return {"report_profile": report, "report_documents": documents, "register_count": register_count,
            "screening": screening, "evidence_findings": findings,
            "disagreements": find_disagreements(findings, report),
            "likelihood": estimate_repeat_likelihood(repeat_model, repeat_features, report, screening),
            "crew_hours": estimate_expected_crew_hours(crew_hours_model, crew_hours_features, report, screening)}


def load_report_context_node(state):
    context = gather_report_context(state["report_reference"])
    run_id, review_id = _identifier(config.RUN_ID_PREFIX), _identifier(config.REVIEW_ID_PREFIX)
    with session_factory() as session:
        insert_agent_run(session, {"run_id": run_id, "status": config.AGENT_RUN_STATUS_RUNNING, "llm_call_count": 0,
                                   **{key: state[key] for key in ("report_reference", "coordinator_request",
                                                                  "coordinator_id", "thread_id")}})
        session.commit()
    return {"run_id": run_id, "review_id": review_id, **context}


def plan_report_review_node(state):
    result = plan_review(state["report_profile"], state["report_documents"], state["screening"],
                         state["register_count"], state["coordinator_request"])
    _record_call(state["run_id"], result["call_record"])
    return {"review_plan": result["review_plan"], "llm_call_count": 1}


def assess_report_requirements_node(state):
    rules = read_policy_section(knowledge_collection, config.POLICY_TRIGGERS_SECTION_NUMBER,
                                config.POLICY_TRIGGERS_SECTION_TITLE)
    result = analyse_requirements(state["screening"], state["report_profile"], state["register_count"],
                                  state["review_plan"], rules)
    _record_call(state["run_id"], result["call_record"])
    return {"requirements_analysis": result["requirements_analysis"], "llm_call_count": 1}


def analyse_report_statements_node(state):
    found = {finding["sentence_type"] for finding in state["evidence_findings"]}
    checks = {sentence_type: read_reviewer_checks(knowledge_collection, sentence_type)
              for sentence_type in config.SENTENCE_TYPE_GUIDE_DOCUMENTS if sentence_type in found}
    result = review_evidence(state["evidence_findings"], state["disagreements"], checks, state["report_profile"],
                             state["screening"], state["review_plan"])
    _record_call(state["run_id"], result["call_record"])
    return {"evidence_review": result["evidence_review"], "llm_call_count": 1}


def interpret_model_predictions_node(state):
    result = analyse_predictions(state["likelihood"], state["crew_hours"], state["report_profile"],
                                 state["screening"], state["review_plan"])
    _record_call(state["run_id"], result["call_record"])
    return {"prediction_analysis": result["prediction_analysis"], "llm_call_count": 1}


def prepare_report_recommendation_node(state):
    rules = read_policy_section(knowledge_collection, config.POLICY_DECISIONS_SECTION_NUMBER,
                                config.POLICY_DECISIONS_SECTION_TITLE)
    result = write_review_recommendation(
        state["report_profile"], state["coordinator_request"], state["screening"], state["register_count"],
        state["likelihood"], state["crew_hours"], state["review_plan"], state["requirements_analysis"],
        state["evidence_review"], state["prediction_analysis"], rules)
    recommendation = result["review_recommendation"]
    write_ups = {"review_plan_text": join_labelled_response(state["review_plan"]),
                 "requirements_analysis_text": join_labelled_response(state["requirements_analysis"]),
                 "evidence_review_text": join_labelled_response(state["evidence_review"]),
                 "prediction_analysis_text": join_labelled_response(state["prediction_analysis"]),
                 "recommendation_text": recommendation["RECOMMENDATION"]}
    _long_and_different(write_ups, "review write-ups")
    profile, screening, register_count = state["report_profile"], state["screening"], state["register_count"]
    review = {"review_id": state["review_id"], "run_id": state["run_id"],
              "report_reference": state["report_reference"], "coordinator_request": state["coordinator_request"],
              **{column: profile[column] for column in (config.REVISION_NUMBER_COLUMN, config.DEFECT_TYPE_COLUMN,
                                                        config.LINKED_DEFECT_TYPE_COLUMN, config.STREET_NAME_COLUMN,
                                                        "district")},
              "requirements": [{key: value for key, value in item.items() if key != "rules"}
                               for item in screening["requirements"]],
              **{column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS},
              **{key: screening[key] for key in ("coordination_triggers", "assessment", "decisions_open")},
              **{key: register_count[key] for key in ("live_permits", "conflicting_permits")},
              "evidence": state["evidence_findings"], "disagreements": state["disagreements"],
              **state["likelihood"], **state["crew_hours"],
              "crew_hours_text": describe_expected_crew_hours(state["crew_hours"]), **write_ups,
              "recommended_decision": recommendation["RECOMMENDED DECISION"], "status": config.REVIEW_STATUS_AWAITING}
    with session_factory() as session:
        insert_llm_call(session, state["run_id"], result["call_record"])
        insert_report_review(session, review)
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_AWAITING_DECISION,
                         state["llm_call_count"] + 1)
        session.commit()
    return {"llm_call_count": 1}


def await_inspector_decision_node(state):
    review_decision = interrupt(state["review_id"])
    refusal = find_decision_refusal(review_decision, state["report_profile"], state["screening"],
                                    state["coordinator_id"])
    if refusal is not None:
        raise ValueError(refusal)
    return {"review_decision": review_decision}


def route_report_decision(state):
    return state["review_decision"]["decision"]


def draft_order_document_node(state):
    decision, profile, screening = state["review_decision"]["decision"], state["report_profile"], state["screening"]
    if decision == config.DECISION_ORDER:
        details = build_works_order_details(profile, screening)
        guidance = _passages(read_works_order_guidance, screening["defect_types"])
    else:
        records = {kind: read_inspection_records(knowledge_collection, kind) for kind in screening["defect_types"]}
        details = build_inspection_request_details(profile, screening, records, state["disagreements"])
        guidance = read_policy_section(knowledge_collection, config.POLICY_DOCUMENTS_SECTION_NUMBER,
                                       config.POLICY_DOCUMENTS_SECTION_TITLE)
    result = draft_order_document(decision, profile, details, guidance)
    _record_call(state["run_id"], result["call_record"])
    return {"order_document": result["order_document"], "document_details": details, "llm_call_count": 1}


def draft_resident_update_node(state):
    guidance = _passages(read_resident_update_guidance, state["screening"]["defect_types"])
    result = draft_resident_update(state["document_details"], guidance)
    _record_call(state["run_id"], result["call_record"])
    return {"resident_update": result["resident_update"], "llm_call_count": 1}


def record_report_outcome_node(state):
    decision = state["review_decision"]
    outcome = {"status": config.REVIEW_STATUS_BY_DECISION[decision["decision"]]}
    if state.get("order_document"):
        outcome.update(order_document_text=join_labelled_response(state["order_document"]),
                       document_details=state["document_details"])
    if state.get("resident_update"):
        outcome["resident_update_text"] = join_labelled_response(state["resident_update"])
    if decision["decision"] == config.DECISION_REFER:
        outcome["coordination_brief"] = build_coordination_brief(
            state["report_profile"], state["screening"], state["register_count"], state["likelihood"],
            state["crew_hours"], state["disagreements"])
    if decision["decision"] == config.DECISION_ORDER:
        _long_and_different({key: outcome.get(key) or "" for key in ("order_document_text", "resident_update_text")},
                            "order texts")
    with session_factory() as session:
        insert_review_decision(session, {"review_id": state["review_id"], "run_id": state["run_id"], **decision})
        record_review_outcome(session, state["review_id"], outcome)
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_COMPLETE, state["llm_call_count"])
        session.commit()
    return {}
