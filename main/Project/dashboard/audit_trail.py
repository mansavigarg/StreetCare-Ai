import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service


def _label(column):
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def _show_brief(brief):
    ui.table(["Answer", "Given"], [[(ui.e(_label(column)[:1].upper() + _label(column)[1:]), ""),
                                    (ui.e(brief[column]), "strong")]
                                   for column in config.COORDINATION_BRIEF_DETAIL_COLUMNS
                                   if column not in config.REPORT_LABEL_COLUMNS])
    st.html('<div style="height:.8rem"></div>')
    st.dataframe(brief["requirements"], hide_index=True)
    permits = [(config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit), "red")
               if permit in brief["conflicting_permits"] else (permit, "gray") for permit in brief["live_permits"]]
    ui.show('<div class="sc-field-label" style="margin-top:.6rem">Coordination triggers</div>'
            + ui.pills([(trigger, "violet") for trigger in brief["coordination_triggers"]]
                       or [(config.CHAT_NONE_TEXT, "gray")])
            + '<div class="sc-field-label" style="margin-top:.8rem">Live permits</div>'
            + ui.pills(permits or [(config.CHAT_NONE_TEXT, "gray")]))
    if brief["flagged_statements"]:
        ui.show("".join(ui.callout(f"<b>{ui.e(flag['document_id'])}</b>: {ui.e(flag['sentence'])} "
                                   f"({ui.e(flag['flag'])})") for flag in brief["flagged_statements"]))


def _show_review(review):
    ui.section(f"{review['report_reference']} · {review['review_id']}")
    ui.show(ui.pills([(config.REPORT_STATUS_LABELS[review["status"]],
                       ui.tone(review["status"], config.STATUS_BADGE_COLOURS)),
                      (f"Assessment: {review['assessment']}",
                       ui.tone(review["assessment"], config.DECISION_BADGE_COLOURS)),
                      (f"Recommended: {review['recommended_decision']}",
                       ui.tone(review["recommended_decision"], config.DECISION_BADGE_COLOURS))]))
    st.html('<div style="height:.8rem"></div>')
    hours_pill = (("Overrun expected", "red") if review["overrun_expected"] else ("No overrun", "gray"))
    ui.kpi_grid([ui.kpi("Repeat likelihood", f"{review['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}",
                        "activity", note_pill=(review["likelihood_band"],
                                               ui.tone(review["likelihood_band"],
                                                       config.LIKELIHOOD_BAND_BADGE_COLOURS))),
                 ui.kpi("Expected crew hours", f"{review['expected_crew_hours']:.{config.CREW_HOURS_DECIMALS}f}",
                        "clock", "blue", note_pill=hours_pill),
                 ui.kpi("Opened by", review["coordinator_id"], "user", "gray", note=review["street_name"])], 3)
    st.html('<div style="height:.9rem"></div>')
    st.caption(f"Expected {review['crew_hours_text']}")
    if review["decision"]:
        ui.show(f'<div class="sc-card"><div class="sc-field-label">Decision</div>'
                + ui.pills([(review["decision"], ui.tone(review["decision"], config.DECISION_BADGE_COLOURS))],
                           meta=f"by {review['decided_by']}, {review['decided_by_role']} for "
                                f"{review['decided_by_district']}")
                + f'<div class="sc-quote">{ui.e(review["decision_reason"])}</div></div>')
        st.html('<div style="height:.8rem"></div>')
    ui.prose_card(review["recommendation_text"], "Recommendation")
    st.html('<div style="height:.8rem"></div>')
    with st.expander("Each agent's write-up"):
        ui.agent_writeups(review)
    if review["order_document_text"]:
        ui.section(config.DOCUMENT_KINDS[review["decision"]].capitalize())
        ui.prose_card(review["order_document_text"])
    if review["resident_update_text"]:
        ui.section("Resident update")
        ui.prose_card(review["resident_update_text"])
    if review["coordination_brief"]:
        ui.section("Coordination brief")
        _show_brief(review["coordination_brief"])
    budget = call_service("GET", f"/reviews/{review['review_id']}/calls")
    seconds = budget["total_latency_milliseconds"] / config.MILLISECONDS_PER_SECOND
    ui.section("Model call budget")
    ui.kpi_grid([ui.kpi("Calls", f"{budget['call_count']} of {budget['expected_call_count']}", "cpu"),
                 ui.kpi("Tokens", f"{budget['total_tokens']:,}", "hash", "blue"),
                 ui.kpi("Latency", f"{seconds:.{config.LATENCY_SECONDS_DECIMALS}f} s", "clock", "orange")], 3)
    st.html('<div style="height:.8rem"></div>')
    st.dataframe([{"#": call["call_id"], "Agent": call["agent_name"].replace("_", " "),
                   "Model": call["model_name"], "Prompt tokens": call["prompt_tokens"],
                   "Response tokens": call["response_tokens"], "Total tokens": call["total_tokens"],
                   "Latency (s)": call["latency_milliseconds"] / config.MILLISECONDS_PER_SECOND}
                  for call in budget["calls"]], hide_index=True,
                 column_config={"Latency (s)": st.column_config.NumberColumn(
                     format=f"%.{config.LATENCY_SECONDS_DECIMALS}f")})


def render_audit_trail_page():
    ui.page_header(config.PAGE_AUDIT, "Every review, decision, document and model call, and every chat turn, as the "
                                      "database recorded it.")
    reviews_tab, chat_tab = st.tabs(["Reviews", "Chat turns"])
    with reviews_tab:
        statuses = st.multiselect("Statuses", config.REVIEW_STATUS_OPTIONS, default=config.DECIDED_REVIEW_STATUSES,
                                  format_func=config.REPORT_STATUS_LABELS.get)
        reviews = call_service("GET", "/reviews", query={"status_value": statuses}) if statuses else []
        table = [{"Report": review["report_reference"], "Revision": review["revision_number"],
                  "Defect type": review["defect_type"],
                  "Linked defect": "—" if review["linked_defect_type"] == config.NO_LINKED_DEFECT
                  else review["linked_defect_type"],
                  "Street": review["street_name"], "Status": config.REPORT_STATUS_LABELS[review["status"]],
                  "Opened by": review["coordinator_id"], "Decided by": review["decided_by"] or "—",
                  "Model calls": review["llm_call_count"]} for review in reviews]
        picked = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row")
        if picked.selection.rows:
            _show_review(reviews[picked.selection.rows[0]])
        elif reviews:
            st.caption("Select a review to see its record, documents and call budget.")
    with chat_tab:
        turns = call_service("GET", "/audit/chat-turns", query={"turn_limit": config.CHAT_AUDIT_MAX_ROWS})
        st.dataframe([{"Turn": turn.get("turn_id"), "Asked by": turn["username"], "Role": turn["user_role"],
                       "Type": turn["question_type"], "Report": turn["report_reference"] or "—",
                       "Street": turn["street_name"] or "—", "Defect type": turn["defect_type"] or "—",
                       "Refused": turn["was_refused"], "Question": turn["question_text"],
                       "Answer": turn["answer_text"], "Sources": turn["source_documents"] or "—"} for turn in turns],
                     hide_index=True, height=min(38 + 35 * len(turns), 640),
                     column_config={"Turn": st.column_config.NumberColumn(width=60),
                                    "Type": st.column_config.TextColumn(width="medium"),
                                    "Role": st.column_config.TextColumn(width="medium"),
                                    "Question": st.column_config.TextColumn(width="large"),
                                    "Answer": st.column_config.TextColumn(width="large")})
