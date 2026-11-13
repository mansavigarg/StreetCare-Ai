from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

from config import config

Base = declarative_base()


class UserAccount(Base):
    __tablename__ = config.USER_ACCOUNTS_TABLE
    username = Column(String, primary_key=True)
    display_name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    district = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    password_salt = Column(String, nullable=False)
    session_token = Column(String, nullable=True, unique=True)


class AgentRun(Base):
    __tablename__ = config.AGENT_RUNS_TABLE
    run_id = Column(String, primary_key=True)
    thread_id = Column(String, nullable=False, unique=True)
    report_reference = Column(String, nullable=False)
    coordinator_id = Column(String, nullable=False)
    coordinator_request = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    llm_call_count = Column(Integer, nullable=False)


class ReportReview(Base):
    __tablename__ = config.REPORT_REVIEWS_TABLE
    review_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False, unique=True)
    report_reference = Column(String, nullable=False)
    revision_number = Column(Integer, nullable=False)
    defect_type = Column(String, nullable=False)
    linked_defect_type = Column(String, nullable=False)
    street_name = Column(String, nullable=False)
    district = Column(String, nullable=True)
    coordinator_request = Column(Text, nullable=False)
    requirements = Column(JSON, nullable=False)
    requirements_met_count = Column(Integer, nullable=False)
    requirements_not_met_count = Column(Integer, nullable=False)
    requirements_not_documented_count = Column(Integer, nullable=False)
    live_permits = Column(JSON, nullable=False)
    conflicting_permits = Column(JSON, nullable=False)
    coordination_triggers = Column(JSON, nullable=False)
    assessment = Column(String, nullable=False)
    decisions_open = Column(JSON, nullable=False)
    evidence = Column(JSON, nullable=False)
    disagreements = Column(JSON, nullable=False)
    repeat_likelihood = Column(Float, nullable=False)
    likelihood_band = Column(String, nullable=False)
    expected_crew_hours = Column(Float, nullable=False)
    planned_crew_hours = Column(Float, nullable=True)
    overrun_expected = Column(Boolean, nullable=True)
    crew_hours_text = Column(Text, nullable=False)
    review_plan_text = Column(Text, nullable=False)
    requirements_analysis_text = Column(Text, nullable=False)
    evidence_review_text = Column(Text, nullable=False)
    prediction_analysis_text = Column(Text, nullable=False)
    recommendation_text = Column(Text, nullable=False)
    recommended_decision = Column(String, nullable=False)
    status = Column(String, nullable=False)
    order_document_text = Column(Text, nullable=True)
    document_details = Column(JSON, nullable=True)
    resident_update_text = Column(Text, nullable=True)
    coordination_brief = Column(JSON, nullable=True)


class ReviewDecision(Base):
    __tablename__ = config.REVIEW_DECISIONS_TABLE
    decision_id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String, nullable=False, unique=True)
    run_id = Column(String, nullable=False)
    decision = Column(String, nullable=False)
    decided_by = Column(String, nullable=False)
    decided_by_role = Column(String, nullable=False)
    decided_by_district = Column(String, nullable=False)
    decision_reason = Column(Text, nullable=False)


class LlmCall(Base):
    __tablename__ = config.LLM_CALLS_TABLE
    call_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    response_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    latency_milliseconds = Column(Integer, nullable=False)


class ChatTurn(Base):
    __tablename__ = config.CHAT_TURNS_TABLE
    turn_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False)
    user_role = Column(String, nullable=False)
    question_text = Column(Text, nullable=False)
    question_type = Column(String, nullable=False)
    report_reference = Column(String, nullable=True)
    street_name = Column(String, nullable=True)
    defect_type = Column(String, nullable=True)
    answer_text = Column(Text, nullable=False)
    source_documents = Column(Text, nullable=False)
    closest_distance = Column(Float, nullable=False)
    was_refused = Column(Boolean, nullable=False)
