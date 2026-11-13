import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service


def _label(column):
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def _sentence_case(text):
    return text[:1].upper() + text[1:]


def _permit_pills(live, conflicting):
    if not live:
        return ui.pill("No live permit", "gray")
    return "".join(ui.pill(config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit), "red")
                   if permit in conflicting else ui.pill(permit, "gray") for permit in live)


def _defect_types(report):
    linked = report[config.LINKED_DEFECT_TYPE_COLUMN]
    return report[config.DEFECT_TYPE_COLUMN] + ("" if linked == config.NO_LINKED_DEFECT else f", {linked}")


def _headline(detail, total):
    report, screening, count = detail["report"], detail["screening"], detail["register_count"]
    likelihood, hours = detail["likelihood"], detail["crew_hours"]
    status = detail["report_status"]
    ui.show(ui.pills([(f"Assessment: {screening['assessment']}",
                       ui.tone(screening["assessment"], config.DECISION_BADGE_COLOURS)),
                      (config.REPORT_STATUS_LABELS[status], ui.tone(status, config.STATUS_BADGE_COLOURS)),
                      (f"Position {detail['queue_position']} of {total}", "gray"),
                      (report["report_channel"], "gray")]))
    st.html('<div style="height:.6rem"></div>')
    standards = sorted({item["standard_document"] for item in screening["requirements"]})
    ui.fields_card([("Street", ui.e(report[config.STREET_NAME_COLUMN]), False),
                    ("District", ui.e(report["district"]), False),
                    ("Street category", ui.e(report[config.STREET_CATEGORY_COLUMN]), False),
                    ("Revision", ui.e(report[config.REVISION_NUMBER_COLUMN]), False),
                    ("Defect type", ui.e(_defect_types(report)), False),
                    ("Reported by", ui.e(report.get("reporter_type", config.NOT_GIVEN_TEXT)), False),
                    ("Where on the street", ui.e(report[config.LOCATION_DESCRIPTION_COLUMN]), True),
                    ("Screened under", "<br>".join(ui.e(name) for name in standards) or "—", True),
                    (f"Street works permits ({count[config.LIVE_STREET_WORKS_COLUMN]} live, "
                     f"{count[config.CONFLICTING_STREET_WORKS_COLUMN]} conflicting)",
                     f'<div class="sc-pills">{_permit_pills(count["live_permits"], count["conflicting_permits"])}'
                     "</div>", True)])
    st.html('<div style="height:1rem"></div>')
    total_requirements = len(screening["requirements"])
    result_cards = {config.REQUIREMENTS_MET_COUNT_COLUMN: ("Requirements met", "check", "green"),
                    config.REQUIREMENTS_NOT_MET_COUNT_COLUMN: ("Not met", "x", "red"),
                    config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: ("Not documented", "help", "orange")}
    triggers = screening["coordination_triggers"]
    planned = hours["planned_crew_hours"]
    planned = config.NOT_GIVEN_TEXT if planned is None else f"{planned:.{config.CREW_HOURS_DECIMALS}f}"
    hours_pill = ((f"Overrun expected · {planned} planned", "red") if hours["overrun_expected"]
                  else (f"On time · {planned} planned", "gray"))
    cards = [ui.kpi(label, screening[column], icon_name, colour, note=f"of {total_requirements} requirements")
             for column, (label, icon_name, colour) in result_cards.items()]
    cards += [ui.kpi("Coordination triggers", len(triggers), "zap", "violet",
                     note=", ".join(triggers) or "None raised"),
              ui.kpi("Repeat likelihood", f"{likelihood['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}",
                     "activity", "primary",
                     note_pill=(likelihood["likelihood_band"],
                                ui.tone(likelihood["likelihood_band"], config.LIKELIHOOD_BAND_BADGE_COLOURS))),
              ui.kpi("Expected crew hours", f"{hours['expected_crew_hours']:.{config.CREW_HOURS_DECIMALS}f}",
                     "clock", "blue", note_pill=hours_pill)]
    ui.kpi_grid(cards, 3)


def _screening_tab(detail):
    screening = detail["screening"]
    several_types = len(screening["defect_types"]) > 1
    rows = []
    for item in screening["requirements"]:
        name = ui.e(item["requirement_name"])
        if several_types:
            name += f'<span class="sc-sub">{ui.e(item["defect_type"])} standard</span>'
        blanks = ", ".join(_label(column) for column in item["blank_answers"]) or "—"
        rows.append([(name, "strong"),
                     (ui.pill(item["result"], ui.tone(item["result"], config.REQUIREMENT_RESULT_BADGE_COLOURS)), ""),
                     (ui.e(item["requirement_sentence"]), ""),
                     (ui.e(blanks), "dim")])
    ui.table(["Requirement", "Result", "What the standard requires", "Answers left blank"], rows)
    ui.section("Decisions open", "The decisions this screening leaves to a Highways Inspector.")
    ui.show(ui.pills([(decision, ui.tone(decision, config.DECISION_BADGE_COLOURS))
                      for decision in screening["decisions_open"]]))
    ui.section("Coordination triggers", "Any one of these sends the report to network coordination.")
    ui.show(ui.pills([(trigger, "violet") for trigger in screening["coordination_triggers"]]
                     or [("No coordination trigger", "gray")]))


def _evidence_tab(detail):
    if not detail["report_documents"]:
        st.info(config.CHAT_NO_DOCUMENTS_TEXT.capitalize())
    flags = {flag["sentence"]: flag for flag in detail["disagreements"]}
    for document in detail["report_documents"]:
        findings = [finding for finding in detail["evidence_findings"]
                    if finding["document_id"] == document[config.DOCUMENT_ID_COLUMN]]
        body = ""
        for finding in findings:
            borne = ", ".join(item["requirement_name"] for item in finding["requirements_borne_on"])
            flag = flags.get(finding["sentence"])
            warning = ui.callout(f"<b>{ui.e(flag['flag'])}</b>: the document states "
                                 f"{ui.e(flag['stated_readings'])}, the report gives "
                                 f"{ui.e(flag['report_readings'])}") if flag else ""
            body += (f'<div class="sc-sentence{" flagged" if flag else ""}">'
                     f'{ui.pill(finding["sentence_type"], config.SENTENCE_TYPE_BADGE_COLOUR)}'
                     f'<div class="sc-sentence-text">{ui.e(finding["sentence"])}</div>'
                     f'<div class="sc-sentence-note">Bears on {ui.e(borne or "no requirement")}</div>'
                     f"{warning}</div>")
        ui.show(f'<div class="sc-doc"><div class="sc-doc-head">'
                f'<span class="sc-icon">{ui.svg("file", "muted", 20)}</span>'
                f'<span class="sc-doc-title">{ui.e(document[config.DOCUMENT_TITLE_COLUMN])}</span>'
                f'<span class="sc-mono sc-muted">{ui.e(document[config.DOCUMENT_ID_COLUMN])}</span>'
                f'{ui.pill(document[config.DOCUMENT_TYPE_COLUMN], "gray")}'
                f'<span class="sc-doc-count">{len(findings)} sentences</span></div>{body}</div>')


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
    rows = []
    for column, purposes in uses.items():
        given = report.get(column, config.NOT_GIVEN_TEXT)
        rows.append([(ui.e(_sentence_case(_label(column))), ""),
                     (ui.e(given), "dim" if given == config.NOT_GIVEN_TEXT else "strong"),
                     (ui.e(_sentence_case(", ".join(dict.fromkeys(purposes)))), "dim")])
    ui.table(["Question", "Answer", "Used for"], rows)


def _review_tab(detail, review, reference):
    if review:
        ui.show(ui.pills([(f"Recommended: {review['recommended_decision']}",
                           ui.tone(review["recommended_decision"], config.DECISION_BADGE_COLOURS)),
                          (config.REPORT_STATUS_LABELS[review["status"]],
                           ui.tone(review["status"], config.STATUS_BADGE_COLOURS))],
                         meta=f"{review['review_id']} · Opened by {review['coordinator_id']}"))
        st.html('<div style="height:.7rem"></div>')
        ui.prose_card(review["recommendation_text"], "Recommendation")
        if review["decision"]:
            st.html('<div style="height:.8rem"></div>')
            decider = (f"by {review['decided_by']}, {review['decided_by_role']} for "
                       f"{review['decided_by_district']}")
            decision_pills = ui.pills([(review["decision"],
                                        ui.tone(review["decision"], config.DECISION_BADGE_COLOURS))], meta=decider)
            ui.show(f'<div class="sc-card"><div class="sc-field-label">Decision</div>{decision_pills}'
                    f'<div class="sc-quote">{ui.e(review["decision_reason"])}</div></div>')
        return
    user, street = st.session_state["signed_in_user"], detail["report"][config.STREET_NAME_COLUMN]
    if user["role"] == config.ROLE_HIGHWAYS_INSPECTOR and street not in config.INSPECTION_DISTRICT_STREETS[
            user["district"]]:
        district = next(name for name, streets in config.INSPECTION_DISTRICT_STREETS.items() if street in streets)
        st.info(f"{street} belongs to the {district} district")
        return
    ui.section("Start a review", f"The review makes {config.LLM_CALLS_BEFORE_DECISION} language model calls and "
                                 "then waits for a Highways Inspector's decision.")
    with st.form("start_review"):
        request = st.text_area("What should the review look at?", placeholder="For example: check the depth "
                               "against the footway standard and any live street works",
                               max_chars=config.COORDINATOR_REQUEST_MAXIMUM_CHARACTERS)
        if st.form_submit_button("Start the review", type="primary", icon=":material/play_arrow:"):
            if len(request.strip()) < config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS:
                st.warning(f"Write at least {config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS} characters")
                return
            call_service("POST", "/reviews", payload={"report_reference": reference, "coordinator_request": request},
                         waiting_message=f"Making {config.LLM_CALLS_BEFORE_DECISION} language model calls",
                         failure_message=config.REVIEW_FAILED_MESSAGE)
            st.rerun()


def render_report_detail_page():
    ui.page_header(config.PAGE_DETAIL, "The screening, statement evidence, form answers and review of one report, "
                                       "before any model call is spent on it.")
    queue = call_service("GET", "/queue", query={"defect_type": config.QUEUE_DEFECT_TYPE_ALL})
    labels = {row["report_reference"]: f"{row['report_reference']}  ·  {row['defect_type']}"
              + ("" if row["linked_defect_type"] == config.NO_LINKED_DEFECT else f", {row['linked_defect_type']}")
              + f"  ·  {row['street_name']}" for row in queue}
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
    _headline(detail, len(queue))
    st.html('<div style="height:1.2rem"></div>')
    screening_tab, evidence_tab, report_tab, review_tab = st.tabs(
        ["Screening", f"Statement evidence  {len(detail['evidence_findings'])}", "Form answers", "Review"])
    with screening_tab:
        _screening_tab(detail)
    with evidence_tab:
        _evidence_tab(detail)
    with report_tab:
        _report_tab(detail)
    with review_tab:
        _review_tab(detail, review, reference)
