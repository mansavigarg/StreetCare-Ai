import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service


def _show_review(review):
    standards = sorted({item["standard_document"] for item in review["requirements"]})
    ui.show(ui.pills([(f"Recommended: {review['recommended_decision']}",
                       ui.tone(review["recommended_decision"], config.DECISION_BADGE_COLOURS)),
                      (f"Assessment: {review['assessment']}",
                       ui.tone(review["assessment"], config.DECISION_BADGE_COLOURS))],
                     meta=" · ".join([review["review_id"], *standards, f"Opened by {review['coordinator_id']}"])))
    st.html('<div style="height:.8rem"></div>')
    planned = review["planned_crew_hours"]
    planned = config.NOT_GIVEN_TEXT if planned is None else f"{planned:.{config.CREW_HOURS_DECIMALS}f}"
    hours_pill = ((f"Overrun expected · {planned} planned", "red") if review["overrun_expected"]
                  else (f"On time · {planned} planned", "gray"))
    triggers = review["coordination_triggers"]
    ui.kpi_grid([
        ui.kpi("Repeat likelihood", f"{review['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}", "activity",
               note_pill=(review["likelihood_band"],
                          ui.tone(review["likelihood_band"], config.LIKELIHOOD_BAND_BADGE_COLOURS))),
        ui.kpi("Expected crew hours", f"{review['expected_crew_hours']:.{config.CREW_HOURS_DECIMALS}f}", "clock",
               "blue", note_pill=hours_pill),
        ui.kpi("Coordination triggers", len(triggers), "zap", "violet", note=", ".join(triggers) or "None raised"),
        ui.kpi("Flagged sentences", len(review["disagreements"]), "flag", "orange",
               note="Statement and report disagree" if review["disagreements"] else "None")], 4)
    ui.section("Recommendation")
    ui.prose_card(review["recommendation_text"])
    ui.section("Requirements")
    rows = [[(ui.e(item["requirement_name"]) + f'<span class="sc-sub">{ui.e(item["defect_type"])} standard</span>',
              "strong"),
             (ui.pill(item["result"], ui.tone(item["result"], config.REQUIREMENT_RESULT_BADGE_COLOURS)), ""),
             (ui.e(", ".join(config.FORM_ANSWER_LABELS.get(column, column) for column in item["blank_answers"])
                   or "—"), "dim")] for item in review["requirements"]]
    ui.table(["Requirement", "Result", "Answers left blank"], rows)
    ui.section("Street works permits live on the street")
    live = review["live_permits"]
    ui.show(ui.pills([(config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit), "red")
                      if permit in review["conflicting_permits"] else (permit, "gray") for permit in live]
                     or [("No live permit", "gray")]))
    if review["disagreements"]:
        ui.section("Flagged statement sentences")
        ui.show("".join(ui.callout(f"<b>{ui.e(flag['document_id'])}</b>: {ui.e(flag['sentence'])} "
                                   f"({ui.e(flag['flag'])})") for flag in review["disagreements"]))
    st.html('<div style="height:1rem"></div>')
    with st.expander("Coordinator's request and each agent's write-up"):
        ui.agent_writeups(review, review["coordinator_request"])


def render_decision_queue_page():
    ui.page_header(config.PAGE_DECISION, "Reviews in your district waiting for a Highways Inspector's decision.")
    user = st.session_state["signed_in_user"]
    streets = config.INSPECTION_DISTRICT_STREETS[user["district"]]
    ui.show(f'<div class="sc-card"><div class="sc-kpi-head" style="margin-bottom:.7rem">{ui.icon("map")}'
            f'<span style="color:var(--sc-ink);font-weight:600">{ui.e(user["district"])} district</span>'
            f'<span>· {len(streets)} streets</span></div>{ui.pills([(street, "gray") for street in streets])}</div>')
    st.html('<div style="height:1rem"></div>')
    recorded = st.session_state.pop("decision_recorded", None)
    if recorded:
        st.success(f"{recorded['report_reference']} is now {config.REPORT_STATUS_PHRASES[recorded['status']]}: "
                   f"{config.DECISION_DESCRIPTIONS[recorded['decision']]}")
    waiting = [review for review in call_service("GET", "/reviews",
                                                 query={"status_value": [config.REVIEW_STATUS_AWAITING]})
               if review["street_name"] in streets]
    chosen = st.session_state.get("chosen_review")
    if chosen and chosen not in [review["review_id"] for review in waiting] and not recorded:
        st.info(f"Review {chosen} has been decided since it was opened here")
        st.session_state.pop("chosen_review")
    if not waiting:
        st.info("No review in your district is waiting for a decision")
        return
    labels = {review["review_id"]: f"{review['report_reference']}  ·  {review['defect_type']}"
              + ("" if review["linked_defect_type"] == config.NO_LINKED_DEFECT
                 else f", {review['linked_defect_type']}") + f"  ·  {review['street_name']}" for review in waiting}
    review_id = st.selectbox(f"Waiting for a decision ({len(waiting)})", list(labels), format_func=labels.get)
    st.session_state["chosen_review"] = review_id
    review = next(item for item in waiting if item["review_id"] == review_id)
    _show_review(review)
    ui.section("Record the decision", "Your decision and reason are recorded against your name and district.")
    pending_by = review.get("pending_decided_by")
    if pending_by and pending_by != user["username"]:
        st.info(f"{pending_by} has already sent {review['pending_decision']} for this review. "
                "Only that inspector can finish recording it.")
        return
    if review["coordinator_id"] == user["username"]:
        st.info("You opened this review, so the other Highways Inspector appointed for the district must decide it")
        return
    pending = st.session_state.get("pending_decision", {})
    pending = pending if pending.get("review_id") == review_id else {}
    options = review["decisions_open"]
    saved_decision = review.get("pending_decision")
    saved_reason = review.get("pending_decision_reason")
    with st.form("decision"):
        decision = st.radio("Decision", options, horizontal=True,
                            index=options.index(saved_decision or pending.get("decision"))
                            if saved_decision or pending.get("decision") else 0)
        reason = st.text_area("Reason, in your own words", value=saved_reason or pending.get("decision_reason", ""),
                              max_chars=config.DECISION_REASON_MAXIMUM_CHARACTERS)
        if st.form_submit_button("Record the decision", type="primary", icon=":material/gavel:"):
            if len(reason.strip()) < config.DECISION_REASON_MINIMUM_CHARACTERS:
                st.warning(f"Write a reason of at least {config.DECISION_REASON_MINIMUM_CHARACTERS} characters")
                return
            st.session_state["pending_decision"] = {"review_id": review_id, "decision": decision,
                                                    "decision_reason": reason}
            st.session_state["decision_recorded"] = call_service(
                "POST", f"/reviews/{review_id}/decision", payload={"decision": decision, "decision_reason": reason},
                waiting_message="Recording the decision and writing what it calls for",
                failure_message=config.DECISION_FAILED_MESSAGE)
            st.session_state.pop("pending_decision")
            st.rerun()
