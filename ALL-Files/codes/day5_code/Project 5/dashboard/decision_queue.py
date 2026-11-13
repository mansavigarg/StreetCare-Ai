import streamlit as st

from config import config
from dashboard.api_client import call_service


def _badge(text, colours):
    st.badge(str(text), color=colours.get(text, config.NEUTRAL_BADGE_COLOUR))


def _show_review(review):
    st.write(f"Recommended by the review opened by {review['coordinator_id']}:")
    _badge(review["recommended_decision"], config.DECISION_BADGE_COLOURS)
    st.write(f"Repeat likelihood {review['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}")
    _badge(review["likelihood_band"], config.LIKELIHOOD_BAND_BADGE_COLOURS)
    hours = f"Expected {review['crew_hours_text']}"
    st.markdown(f":{config.OVERRUN_COLOUR}[{hours}]" if review["overrun_expected"] else hours)
    st.write(review["recommendation_text"])
    for item in review["requirements"]:
        st.write(f"{item['defect_type']} standard: {item['requirement_name']}")
        _badge(item["result"], config.REQUIREMENT_RESULT_BADGE_COLOURS)
        if item["blank_answers"]:
            st.caption("Left blank: " + ", ".join(config.FORM_ANSWER_LABELS.get(column, column)
                                                  for column in item["blank_answers"]))
    st.write("Coordination triggers: " + (", ".join(review["coordination_triggers"]) or config.CHAT_NONE_TEXT))
    permits = [config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit)
               if permit in review["conflicting_permits"] else permit for permit in review["live_permits"]]
    st.write("Live permits: " + (", ".join(permits) or config.CHAT_NONE_TEXT))
    for flag in review["disagreements"]:
        st.warning(f"{flag['document_id']}: {flag['sentence']} ({flag['flag']})")
    with st.expander("What was asked and each agent's write-up"):
        st.write(review["coordinator_request"])
        for key in ("review_plan_text", "requirements_analysis_text", "evidence_review_text",
                    "prediction_analysis_text"):
            st.text(review[key])


def render_decision_queue_page():
    user = st.session_state["signed_in_user"]
    streets = config.INSPECTION_DISTRICT_STREETS[user["district"]]
    st.write(f"**{user['district']}**: {', '.join(streets)}")
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
    labels = {review["review_id"]: f"{review['report_reference']} | {review['defect_type']}"
              + ("" if review["linked_defect_type"] == config.NO_LINKED_DEFECT
                 else f", {review['linked_defect_type']}") + f" | {review['street_name']}" for review in waiting}
    review_id = st.selectbox("Review", list(labels), format_func=labels.get)
    st.session_state["chosen_review"] = review_id
    review = next(item for item in waiting if item["review_id"] == review_id)
    _show_review(review)
    if review["coordinator_id"] == user["username"]:
        st.info("You opened this review, so the other Highways Inspector appointed for the district must decide it")
        return
    pending = st.session_state.get("pending_decision", {})
    pending = pending if pending.get("review_id") == review_id else {}
    options = review["decisions_open"]
    with st.form("decision"):
        decision = st.radio("Decision", options, index=options.index(pending["decision"]) if pending else 0)
        reason = st.text_area("Reason", value=pending.get("decision_reason", ""),
                              max_chars=config.DECISION_REASON_MAXIMUM_CHARACTERS)
        if st.form_submit_button("Record the decision"):
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
