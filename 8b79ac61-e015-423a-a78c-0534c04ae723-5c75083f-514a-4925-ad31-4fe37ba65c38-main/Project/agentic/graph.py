import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic.nodes import (analyse_report_statements_node, assess_report_requirements_node,
                           await_inspector_decision_node, draft_order_document_node, draft_resident_update_node,
                           interpret_model_predictions_node, load_report_context_node, plan_report_review_node,
                           prepare_report_recommendation_node, record_report_outcome_node, route_report_decision)
from agentic.state import ReportReviewState
from config import config

REVIEW_STEPS = [("load_report_context", load_report_context_node), ("plan_report_review", plan_report_review_node),
                ("assess_report_requirements", assess_report_requirements_node),
                ("analyse_report_statements", analyse_report_statements_node),
                ("interpret_model_predictions", interpret_model_predictions_node),
                ("prepare_report_recommendation", prepare_report_recommendation_node),
                ("await_inspector_decision", await_inspector_decision_node)]


def build_workflow(checkpoint_path):
    builder = StateGraph(ReportReviewState)
    for name, node in REVIEW_STEPS + [("draft_order_document", draft_order_document_node),
                                      ("draft_resident_update", draft_resident_update_node),
                                      ("record_report_outcome", record_report_outcome_node)]:
        builder.add_node(name, node)
    names = [START] + [name for name, _ in REVIEW_STEPS]
    for source, target in zip(names, names[1:]):
        builder.add_edge(source, target)
    builder.add_conditional_edges("await_inspector_decision", route_report_decision, {
        config.DECISION_ORDER: "draft_order_document", config.DECISION_INSPECT: "draft_order_document",
        config.DECISION_REFER: "record_report_outcome"})
    builder.add_conditional_edges("draft_order_document", route_report_decision, {
        config.DECISION_ORDER: "draft_resident_update", config.DECISION_INSPECT: "record_report_outcome"})
    builder.add_edge("draft_resident_update", "record_report_outcome")
    builder.add_edge("record_report_outcome", END)
    checkpointer = SqliteSaver(sqlite3.connect(checkpoint_path, check_same_thread=False))
    return builder.compile(checkpointer=checkpointer)
