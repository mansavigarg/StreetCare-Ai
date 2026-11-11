from sqlalchemy import select

from persistence.models import AgentRun, ChatTurn, LlmCall, ReportReview, ReviewDecision, UserAccount


def _record(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _one(session, model, key_column, key_value):
    row = session.get(model, key_value)
    if row is None:
        raise ValueError(f"No {model.__tablename__} row has {key_column} {key_value}")
    return row


def insert_user_account(session, account_record):
    session.add(UserAccount(**account_record))


def fetch_user_account(session, condition):
    row = session.scalars(select(UserAccount).where(condition)).first()
    return None if row is None else _record(row)


def update_session_token(session, username, session_token):
    _one(session, UserAccount, "username", username).session_token = session_token


def insert_agent_run(session, run_record):
    session.add(AgentRun(**run_record))


def update_agent_run(session, run_id, status_value, llm_call_count):
    run = _one(session, AgentRun, "run_id", run_id)
    run.status, run.llm_call_count = status_value, llm_call_count


def insert_llm_call(session, run_id, call_record):
    session.add(LlmCall(run_id=run_id, **call_record))


def fetch_llm_calls(session, run_id):
    rows = session.scalars(select(LlmCall).where(LlmCall.run_id == run_id).order_by(LlmCall.call_id))
    return [_record(row) for row in rows]


def insert_report_review(session, review_record):
    session.add(ReportReview(**review_record))


def fetch_review_records(session, condition):
    query = (select(ReportReview, AgentRun, ReviewDecision)
             .join(AgentRun, AgentRun.run_id == ReportReview.run_id)
             .outerjoin(ReviewDecision, ReviewDecision.review_id == ReportReview.review_id)
             .where(condition)
             .order_by(ReportReview.report_reference, ReportReview.revision_number, ReportReview.review_id))
    records = []
    for review, run, decision in session.execute(query):
        record = _record(review)
        record.update({field: getattr(run, field) for field in ("thread_id", "coordinator_id", "llm_call_count")})
        for field in ("decision", "decided_by", "decided_by_role", "decided_by_district", "decision_reason"):
            record[field] = None if decision is None else getattr(decision, field)
        records.append(record)
    return records


def insert_review_decision(session, decision_record):
    session.add(ReviewDecision(**decision_record))


def record_review_outcome(session, review_id, outcome_record):
    review = _one(session, ReportReview, "review_id", review_id)
    for column, value in outcome_record.items():
        setattr(review, column, value)


def insert_chat_turn(session, turn_record):
    session.add(ChatTurn(**turn_record))


def fetch_chat_turns(session, condition, turn_limit):
    rows = session.scalars(select(ChatTurn).where(condition).order_by(ChatTurn.turn_id.desc()).limit(turn_limit))
    return [_record(row) for row in rows]
