from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from config import config

DefectTypeFilter = Literal[tuple(config.QUEUE_DEFECT_TYPE_OPTIONS)]
ReviewStatusValue = Literal[tuple(config.REVIEW_STATUS_OPTIONS)]
DecisionValue = Literal[tuple(config.DECISIONS)]
CoordinatorRequestText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS,
    max_length=config.COORDINATOR_REQUEST_MAXIMUM_CHARACTERS)]
DecisionReasonText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.DECISION_REASON_MINIMUM_CHARACTERS,
    max_length=config.DECISION_REASON_MAXIMUM_CHARACTERS)]
QuestionText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.CHAT_QUESTION_MINIMUM_CHARACTERS,
    max_length=config.CHAT_QUESTION_MAXIMUM_CHARACTERS)]


class SignInRequest(BaseModel):
    username: str
    password: str


class SignedInUserResponse(BaseModel):
    username: str
    display_name: str
    role: str
    district: str | None
    session_token: str
    pages: list[str]
    permissions: list[str]


class SignedOutResponse(BaseModel):
    username: str
    signed_out: bool


class QueueRow(BaseModel):
    queue_position: int
    report_reference: str
    revision_number: int
    defect_type: str
    linked_defect_type: str
    street_name: str
    district: str
    report_channel: str
    report_status: str
    document_count: int
    requirements_met_count: int
    requirements_not_met_count: int
    requirements_not_documented_count: int
    live_permit_count: int
    conflicting_permit_count: int
    coordination_trigger_count: int
    assessment: str
    repeat_likelihood: float
    likelihood_band: str
    expected_crew_hours: float
    overrun_expected: bool | None
    disagreement_count: int


class ReportDetailResponse(BaseModel):
    queue_position: int
    report_status: str
    report: dict
    report_documents: list[dict]
    screening: dict
    register_count: dict
    evidence_findings: list[dict]
    disagreements: list[dict]
    likelihood: dict
    crew_hours: dict


class ReviewRequest(BaseModel):
    report_reference: str
    coordinator_request: CoordinatorRequestText


class DecisionRequest(BaseModel):
    decision: DecisionValue
    decision_reason: DecisionReasonText


class ReviewRecordResponse(BaseModel):
    review_id: str
    run_id: str
    thread_id: str
    coordinator_id: str
    report_reference: str
    revision_number: int
    defect_type: str
    linked_defect_type: str
    street_name: str
    district: str | None
    coordinator_request: str
    requirements: list[dict]
    requirements_met_count: int
    requirements_not_met_count: int
    requirements_not_documented_count: int
    live_permits: list[str]
    conflicting_permits: list[str]
    coordination_triggers: list[str]
    assessment: str
    decisions_open: list[str]
    evidence: list[dict]
    disagreements: list[dict]
    repeat_likelihood: float
    likelihood_band: str
    expected_crew_hours: float
    planned_crew_hours: float | None
    overrun_expected: bool | None
    crew_hours_text: str
    review_plan_text: str
    requirements_analysis_text: str
    evidence_review_text: str
    prediction_analysis_text: str
    recommendation_text: str
    recommended_decision: str
    status: str
    order_document_text: str | None
    document_details: dict | None
    resident_update_text: str | None
    coordination_brief: dict | None
    llm_call_count: int
    decision: str | None
    decided_by: str | None
    decided_by_role: str | None
    decided_by_district: str | None
    decision_reason: str | None


class ReviewCallsResponse(BaseModel):
    review_id: str
    run_id: str
    call_count: int
    expected_call_count: int
    total_tokens: int
    total_latency_milliseconds: int
    calls: list[dict]


class ChatRequest(BaseModel):
    question: QuestionText


class ChatResponse(BaseModel):
    question_type: str
    report_reference: str | None
    street_name: str | None
    defect_type: str | None
    answer: str
    source_documents: list[str]
    was_refused: bool


class ChatTurnResponse(BaseModel):
    turn_id: int
    username: str
    user_role: str
    question_text: str
    question_type: str
    report_reference: str | None
    street_name: str | None
    defect_type: str | None
    answer_text: str
    source_documents: str
    closest_distance: float
    was_refused: bool
