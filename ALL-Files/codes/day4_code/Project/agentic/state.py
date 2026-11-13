import operator
from typing import Annotated, TypedDict


class ReportReviewState(TypedDict):
    report_reference: str
    coordinator_request: str
    coordinator_id: str
    thread_id: str
    run_id: str
    review_id: str
    report_profile: dict
    report_documents: list[dict]
    register_count: dict
    screening: dict
    evidence_findings: list[dict]
    disagreements: list[dict]
    likelihood: dict
    crew_hours: dict
    review_plan: dict
    requirements_analysis: dict
    evidence_review: dict
    prediction_analysis: dict
    review_decision: dict
    order_document: dict
    document_details: dict
    resident_update: dict
    llm_call_count: Annotated[int, operator.add]
