import streamlit as st

from config import config
from dashboard.api_client import call_service


def _badge(text, colours):
    st.badge(str(text), color=colours.get(text, config.NEUTRAL_BADGE_COLOUR))


def _label(column):
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def _permits(register_count):
    return [config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit)
            if permit in register_count["conflicting_permits"] else permit
            for permit in register_count["live_permits"]] or [config.CHAT_NONE_TEXT]


def _headline(detail):
    report, screening, count = detail["report"], detail["screening"], detail["register_count"]
    likelihood, hours = detail["likelihood"], detail["crew_hours"]
    _badge(screening["assessment"], config.DECISION_BADGE_COLOURS)
    st.badge(config.REPORT_STATUS_LABELS[detail["report_status"]],
             color=config.STATUS_BADGE_COLOURS[detail["report_status"]])
    figures = [("Queue position", detail["queue_position"]), ("District", report["district"]),
               ("Channel", report["report_channel"]), ("Revision", report[config.REVISION_NUMBER_COLUMN]),
               *[(_label(column).capitalize(), screening[column]) for column in config.REQUIREMENT_COUNT_RESULTS],
               ("Triggers", len(screening["coordination_triggers"])),
               ("Live permits", count[config.LIVE_STREET_WORKS_COLUMN]),
               ("Conflicting permits", count[config.CONFLICTING_STREET_WORKS_COLUMN])]
    for column, (label, value) in zip(st.columns(len(figures)), figures):
        column.metric(label, value)
    st.write(f"Repeat likelihood {likelihood['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}")
    _badge(likelihood["likelihood_band"], config.LIKELIHOOD_BAND_BADGE_COLOURS)
    planned = hours["planned_crew_hours"]
    planned = config.NOT_GIVEN_TEXT if planned is None else f"{planned:.{config.CREW_HOURS_DECIMALS}f}"
    text = f"Expected crew hours {hours['expected_crew_hours']:.{config.CREW_HOURS_DECIMALS}f} against {planned} planned"
    st.markdown(f":{config.OVERRUN_COLOUR}[{text}, overrun expected]" if hours["overrun_expected"] else text)


def _screening_tab(detail):
    screening = detail["screening"]
    for item in screening["requirements"]:
        st.markdown(f"**{item['defect_type']} standard: {item['requirement_name']}**")
        _badge(item["result"], config.REQUIREMENT_RESULT_BADGE_COLOURS)
        st.caption(item["requirement_sentence"])
        if item["blank_answers"]:
            st.write("Left blank: " + ", ".join(_label(column) for column in item["blank_answers"]))
    st.write("Decisions open: " + ", ".join(screening["decisions_open"]))
    st.write("Coordination triggers: " + (", ".join(screening["coordination_triggers"]) or config.CHAT_NONE_TEXT))
    st.write("Live permits: " + ", ".join(_permits(detail["register_count"])))


def _evidence_tab(detail):
    if not detail["report_documents"]:
        st.info(config.CHAT_NO_DOCUMENTS_TEXT.capitalize())
    flags = {flag["sentence"]: flag for flag in detail["disagreements"]}
    for document in detail["report_documents"]:
        st.markdown(f"**{document[config.DOCUMENT_ID_COLUMN]}: {document[config.DOCUMENT_TITLE_COLUMN]}** "
                    f"({document[config.DOCUMENT_TYPE_COLUMN]})")
        for finding in detail["evidence_findings"]:
            if finding["document_id"] != document[config.DOCUMENT_ID_COLUMN]:
                continue
            st.badge(finding["sentence_type"], color=config.SENTENCE_TYPE_BADGE_COLOUR)
            st.write(finding["sentence"])
            st.caption("Bears on: " + (", ".join(item["requirement_name"] for item in finding["requirements_borne_on"])
                                       or config.CHAT_NONE_TEXT))
            flag = flags.get(finding["sentence"])
            if flag:
                st.warning(f"{flag['flag']}: the document states {flag['stated_readings']}, the report gives "
                           f"{flag['report_readings']}")


def _report_tab(detail):
    report, screening = detail["report"], detail["screening"]
    uses = {}
    for column in config.WORKS_ORDER_DETAIL_COLUMNS:
        if column not in config.REPORT_LABEL_COLUMNS:
            uses.setdefault(column, []).append(config.DOCUMENT_KINDS[config.DECISION_ORDER])
    for item in screening["requirements"]:
        for column in config.REQUIREMENT_ANSWER_COLUMNS[item["requirement_kind"]]:
            uses.setdefault(column, []).append(f"{item['defect_type']} {item['requirement_name']}")
    for defect_type in screening["defect_types"]:
        for column in config.WORKS_ORDER_MEASUREMENT_COLUMNS[defect_type]:
            uses.setdefault(column, []).append(f"{defect_type} measurement")
    for trigger, rule in config.TRIGGER_FORM_ANSWERS.items():
        if set(rule["defect_types"]) & set(screening["defect_types"]):
            uses.setdefault(rule["answer_column"], []).append(trigger)
    st.dataframe([{"Answer": _label(column), "Given": report.get(column, config.NOT_GIVEN_TEXT),
                   "Used for": ", ".join(dict.fromkeys(purposes))} for column, purposes in uses.items()],
                 hide_index=True)


def _review_tab(detail, review, reference):
    if review:
        st.write(review["recommendation_text"])
        _badge(review["recommended_decision"], config.DECISION_BADGE_COLOURS)
        if review["decision"]:
            st.write(f"Decided {review['decision']} by {review['decided_by']}, {review['decided_by_role']} for "
                     f"{review['decided_by_district']}: {review['decision_reason']}")
        return
    user, street = st.session_state["signed_in_user"], detail["report"][config.STREET_NAME_COLUMN]
    if user["role"] == config.ROLE_HIGHWAYS_INSPECTOR and street not in config.INSPECTION_DISTRICT_STREETS[
            user["district"]]:
        district = next(name for name, streets in config.INSPECTION_DISTRICT_STREETS.items() if street in streets)
        st.info(f"{street} belongs to the {district} district")
        return
    with st.form("start_review"):
        request = st.text_area("What should the review look at?",
                               max_chars=config.COORDINATOR_REQUEST_MAXIMUM_CHARACTERS)
        if st.form_submit_button("Start the review"):
            if len(request.strip()) < config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS:
                st.warning(f"Write at least {config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS} characters")
                return
            call_service("POST", "/reviews", payload={"report_reference": reference, "coordinator_request": request},
                         waiting_message=f"Making {config.LLM_CALLS_BEFORE_DECISION} language model calls",
                         failure_message=config.REVIEW_FAILED_MESSAGE)
            st.rerun()


def render_report_detail_page():
    queue = call_service("GET", "/queue", query={"defect_type": config.QUEUE_DEFECT_TYPE_ALL})
    labels = {row["report_reference"]: f"{row['report_reference']} | {row['defect_type']}"
              + ("" if row["linked_defect_type"] == config.NO_LINKED_DEFECT else f", {row['linked_defect_type']}")
              + f" | {row['street_name']}" for row in queue}
    references = list(labels)
    chosen = st.session_state.get("chosen_report")
    reference = st.selectbox("Report", references, format_func=labels.get,
                             index=references.index(chosen) if chosen in references else 0)
    st.session_state["chosen_report"] = reference
    detail = call_service("GET", f"/queue/{reference}")
    reviews = call_service("GET", "/reviews", query={"status_value": config.REVIEW_STATUS_OPTIONS})
    revision = next(row["revision_number"] for row in queue if row["report_reference"] == reference)
    review = next((record for record in reviews if record["report_reference"] == reference
                   and record["revision_number"] == revision), None)
    _headline(detail)
    screening_tab, evidence_tab, report_tab, review_tab = st.tabs(["Screening", "Statement evidence", "Report",
                                                                   "Review"])
    with screening_tab:
        _screening_tab(detail)
    with evidence_tab:
        _evidence_tab(detail)
    with report_tab:
        _report_tab(detail)
    with review_tab:
        _review_tab(detail, review, reference)
