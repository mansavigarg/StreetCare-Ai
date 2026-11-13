import streamlit as st

from config import config
from dashboard.api_client import call_service


def _label(column):
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def _show_brief(brief):
    st.dataframe([{"Answer": _label(column), "Given": brief[column]}
                  for column in config.COORDINATION_BRIEF_DETAIL_COLUMNS if column not in config.REPORT_LABEL_COLUMNS],
                  hide_index=True)
    st.dataframe(brief["requirements"], hide_index=True)
    st.write("Coordination triggers: " + (", ".join(brief["coordination_triggers"]) or config.CHAT_NONE_TEXT))
    permits = [config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit)
               if permit in brief["conflicting_permits"] else permit for permit in brief["live_permits"]]
    st.write("Live permits: " + (", ".join(permits) or config.CHAT_NONE_TEXT))
    for flag in brief["flagged_statements"]:
        st.warning(f"{flag['document_id']}: {flag['sentence']} ({flag['flag']})")


def _show_review(review):
    st.write(f"Assessment {review['assessment']}, recommended {review['recommended_decision']}")
    st.write(f"Repeat likelihood {review['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f} "
             f"({review['likelihood_band']}); expected {review['crew_hours_text']}")
    if review["decision"]:
        st.write(f"{review['decided_by']}, {review['decided_by_role']} for {review['decided_by_district']}, decided "
                 f"{review['decision']}: {review['decision_reason']}")
    st.write(review["recommendation_text"])
    with st.expander("Each agent's write-up"):
        for key in ("review_plan_text", "requirements_analysis_text", "evidence_review_text",
                    "prediction_analysis_text"):
            st.text(review[key])
    if review["order_document_text"]:
        st.subheader(config.DOCUMENT_KINDS[review["decision"]].capitalize())
        st.text(review["order_document_text"])
    if review["resident_update_text"]:
        st.subheader("Resident update")
        st.text(review["resident_update_text"])
    if review["coordination_brief"]:
        st.subheader("Coordination brief")
        _show_brief(review["coordination_brief"])
    budget = call_service("GET", f"/reviews/{review['review_id']}/calls")
    seconds = budget["total_latency_milliseconds"] / config.MILLISECONDS_PER_SECOND
    figures = [("Calls", f"{budget['call_count']} of {budget['expected_call_count']}"),
               ("Tokens", budget["total_tokens"]), ("Latency (s)", f"{seconds:.{config.LATENCY_SECONDS_DECIMALS}f}")]
    for column, (label, value) in zip(st.columns(len(figures)), figures):
        column.metric(label, value)
    st.dataframe(budget["calls"], hide_index=True)


def render_audit_trail_page():
    reviews_tab, chat_tab = st.tabs(["Reviews", "Chat turns"])
    with reviews_tab:
        statuses = st.multiselect("Statuses", config.REVIEW_STATUS_OPTIONS, default=config.DECIDED_REVIEW_STATUSES,
                                  format_func=config.REPORT_STATUS_LABELS.get)
        reviews = call_service("GET", "/reviews", query={"status_value": statuses}) if statuses else []
        table = [{"Report": review["report_reference"], "Revision": review["revision_number"],
                  "Defect type": review["defect_type"], "Linked defect type": review["linked_defect_type"],
                  "Street": review["street_name"], "Status": config.REPORT_STATUS_LABELS[review["status"]],
                  "Opened by": review["coordinator_id"], "Decided by": review["decided_by"],
                  "Model calls": review["llm_call_count"]} for review in reviews]
        picked = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row")
        if picked.selection.rows:
            _show_review(reviews[picked.selection.rows[0]])
    with chat_tab:
        turns = call_service("GET", "/audit/chat-turns", query={"turn_limit": config.CHAT_AUDIT_MAX_ROWS})
        st.dataframe([{"Asked by": turn["username"], "Role": turn["user_role"], "Type": turn["question_type"],
                       "Report": turn["report_reference"], "Street": turn["street_name"],
                       "Defect type": turn["defect_type"], "Refused": turn["was_refused"],
                       "Question": turn["question_text"], "Answer": turn["answer_text"],
                       "Sources": turn["source_documents"]} for turn in turns], hide_index=True)
