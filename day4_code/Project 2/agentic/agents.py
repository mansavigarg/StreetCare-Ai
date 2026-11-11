import re

from agentic.language_model import call_language_model, parse_labelled_response
from config import config
from tools.prediction_tools import describe_expected_crew_hours
from tools.report_data_tools import describe_record

FIELDS_ONLY = ("Use only the facts given below. Do not work anything out again, do not change any result, and "
               "do not name any figure, person, time, date or contact detail that is not given below.")


def _speaker(task):
    return f"You write for the {config.OFFICE_NAME} of {config.ORGANISATION_NAME}. {task} {FIELDS_ONLY}"


def _label(column):
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def _list(items):
    return ", ".join(str(item) for item in items) or config.CHAT_NONE_TEXT


def _answers(report, columns):
    described = describe_record({column: report.get(column) for column in columns})
    return "\n".join(f"- {_label(column)}: {described[column]}" for column in columns) or config.CHAT_NONE_TEXT


def _report(report):
    return _answers(report, list(report))


def _permits(register_count):
    return (f"Street works permits live on the street: {_list(register_count['live_permits'])}. "
            f"Conflicting with the repair: {_list(register_count['conflicting_permits'])}.")


def _counts(screening):
    return ", ".join(f"{_label(column)}: {screening[column]}" for column in config.REQUIREMENT_COUNT_RESULTS)


def _screening(screening):
    return (f"Coordination triggers that apply: {_list(screening['coordination_triggers'])}.\n"
            f"Assessment: {screening['assessment']}. Decisions open: {_list(screening['decisions_open'])}.")


def _requirements(screening, report=None):
    lines = []
    for item in screening["requirements"]:
        line = (f"- {item['defect_type']} standard ({item['standard_document']}), {item['requirement_name']}: "
                f"{item['requirement_sentence']} Result: {item['result']}. "
                f"Answers left blank: {_list(_label(column) for column in item['blank_answers'])}.")
        if report is not None:
            line += "\n" + _answers(report, config.REQUIREMENT_ANSWER_COLUMNS[item["requirement_kind"]])
        lines.append(line)
    return "\n".join(lines) or config.CHAT_NONE_TEXT


def _layout(labels):
    return ("Reply in exactly this layout, each label in capitals followed by a colon, and nothing else:\n"
            + "\n".join(f"{label}: ..." for label in labels))


def _ask(prompt_parts, labels, agent_name):
    reply = call_language_model("\n\n".join(prompt_parts + [_layout(labels)]), agent_name)
    return parse_labelled_response(reply["response_text"], labels, agent_name), reply["call_record"]


def _flat(text):
    return " ".join(text.lower().split())


def _signed_off(text):
    spaced = lambda words: " ".join(re.sub(config.SIGN_OFF_SPACING_PATTERN, " ", words).lower().split())
    return spaced(text).endswith(spaced(config.DOCUMENT_SIGN_OFF))


def _one_line(fields, label, agent_name):
    if "\n" in fields[label]:
        raise ValueError(f"{agent_name} wrote its {label} over more than one line")


def _refuse_wording(text, slot_text, agent_name):
    named = [term for term in config.DOCUMENT_FORBIDDEN_TERMS if term in _flat(text)]
    slots = [character for character in config.TEMPLATE_SLOT_CHARACTERS if character in slot_text]
    if named or slots:
        raise ValueError(f"{agent_name} wrote {_list(named + slots)}, which its document may not hold")
    if not _signed_off(slot_text):
        raise ValueError(f"{agent_name} did not end its document with {config.DOCUMENT_SIGN_OFF}")


def plan_review(report_profile, report_documents, screening, register_count, coordinator_request):
    documents = [f"- {document[config.DOCUMENT_ID_COLUMN]}: {document[config.DOCUMENT_TITLE_COLUMN]}"
                 for document in report_documents]
    labels = ["PRIORITY", "CHECKS", "RATIONALE"]
    fields, call_record = _ask([
        _speaker("Plan what the review of one street defect report has to establish."),
        "The report:\n" + _report(report_profile),
        "Its documents, in order:\n" + ("\n".join(documents) or config.CHAT_NONE_TEXT),
        "Each requirement of its standards:\n" + _requirements(screening),
        "Counts: " + _counts(screening) + "\n" + _permits(register_count) + "\n" + _screening(screening),
        "What the person who opened the review asked: " + coordinator_request,
        "Never alter a screening result. PRIORITY is what matters most, CHECKS is a numbered list of what the "
        "review must check, RATIONALE is why."], labels, config.AGENT_NAME_PLANNER)
    return {"review_plan": fields, "call_record": call_record}


def analyse_requirements(screening, report_profile, register_count, review_plan, trigger_rules):
    labels = ["REQUIREMENTS MET", "REQUIREMENTS NOT MET", "NOT DOCUMENTED", "COORDINATION TRIGGERS",
              "REQUIREMENTS SUMMARY"]
    fields, call_record = _ask([
        _speaker("Account for each requirement's result under its standard, and for the coordination triggers, in "
                 "words a Highways Inspector can check against the report."),
        "Each requirement, in order, with the answers it reads:\n" + _requirements(screening, report_profile),
        "Counts: " + _counts(screening),
        "What the Highways Policy says about the coordination triggers:\n" + trigger_rules,
        "The answers the triggers read:\n" + _answers(report_profile, config.TRIGGER_ANSWER_COLUMNS),
        _permits(register_count) + "\n" + _screening(screening),
        "The checks the plan sets:\n" + review_plan["CHECKS"],
        "Take every result as given: a blank answer stays blank. Never present the report as decided, rejected or "
        "closed. Write none in a field with nothing to report."], labels, config.AGENT_NAME_REQUIREMENTS)
    return {"requirements_analysis": fields, "call_record": call_record}


def review_evidence(evidence_findings, disagreements, reviewer_checks, report_profile, screening, review_plan):
    columns = [column for item in screening["requirements"]
               for column in config.REQUIREMENT_ANSWER_COLUMNS[item["requirement_kind"]]]
    columns = list(dict.fromkeys(columns + config.STATEMENT_GUIDE_ANSWER_COLUMNS))
    sentences = [f"- {finding['document_id']} ({finding['document_type']}), {finding['sentence_type']}: "
                 f"\"{finding['sentence']}\" Bears on: "
                 f"{_list(item['requirement_name'] for item in finding['requirements_borne_on'])}"
                 for finding in evidence_findings]
    flags = [f"- {flag['document_id']}: \"{flag['sentence']}\" {flag['flag']}. The document states "
             f"{describe_record(flag['stated_readings'])}; the report gives {describe_record(flag['report_readings'])}"
             for flag in disagreements]
    checks = [f"- {sentence_type} ({config.SENTENCE_TYPE_GUIDE_DOCUMENTS[sentence_type]}): {text}"
              for sentence_type, text in reviewer_checks.items()]
    labels = ["EVIDENCE FOUND", "DISAGREEMENTS", "CHECKS TO MAKE", "EVIDENCE SUMMARY"]
    fields, call_record = _ask([
        _speaker("Set the evidence in the report's documents beside the report itself."),
        "The report's answers:\n" + _answers(report_profile, columns),
        "Every sentence of the documents, document by document:\n" + ("\n".join(sentences) or config.CHAT_NONE_TEXT),
        "The sentences flagged against the report:\n" + ("\n".join(flags) or config.CHAT_NONE_TEXT),
        "The checks each statement guide sets:\n" + ("\n".join(checks) or config.CHAT_NONE_TEXT),
        "The checks the plan sets:\n" + review_plan["CHECKS"],
        "A document is evidence only and never stands in for an answer on the report. Flags are left to the "
        "Highways Inspector; any other difference is advice only. Write none in a field with nothing to report."],
        labels, config.AGENT_NAME_EVIDENCE)
    return {"evidence_review": fields, "call_record": call_record}


def analyse_predictions(likelihood, crew_hours, report_profile, screening, review_plan):
    flags = {column: int(trigger in screening["coordination_triggers"])
             for trigger, column in config.COORDINATION_TRIGGER_FLAG_COLUMNS.items()}
    inputs = {**report_profile, **{column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS}, **flags}
    labels = ["LIKELIHOOD", "EXPECTED CREW HOURS", "PREDICTION SUMMARY"]
    fields, call_record = _ask([
        _speaker("Interpret the two model figures for one report, with the inputs behind each."),
        f"Repeat likelihood: {likelihood['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}, band "
        f"{likelihood['likelihood_band']}. The band is {config.HIGH_LIKELIHOOD_BAND} from "
        f"{config.HIGH_LIKELIHOOD_MINIMUM}, {config.MEDIUM_LIKELIHOOD_BAND} from {config.MEDIUM_LIKELIHOOD_MINIMUM} "
        f"and {config.LOW_LIKELIHOOD_BAND} below.",
        "The repeat model's inputs:\n" + _answers(inputs, config.REPEAT_NUMERIC_FEATURES
                                                  + config.REPEAT_CATEGORICAL_FEATURES),
        f"Expected crew hours: {describe_expected_crew_hours(crew_hours)} An overrun means more than "
        f"{config.OVERRUN_RATIO} times the planned hours.",
        "The crew hours model's inputs:\n" + _answers(inputs, config.CREW_HOURS_NUMERIC_FEATURES
                                                      + config.CREW_HOURS_CATEGORICAL_FEATURES),
        f"A blank input is filled with the {config.NUMBER_FILL_STRATEGY} for a number and the "
        f"{config.CATEGORY_FILL_STRATEGY.replace('_', ' ')} answer for a category, from the training data.",
        "The checks the plan sets:\n" + review_plan["CHECKS"],
        "Neither figure decides anything or appears in any document. Never recalculate either."],
        labels, config.AGENT_NAME_RISK)
    return {"prediction_analysis": fields, "call_record": call_record}


def write_review_recommendation(report_profile, coordinator_request, screening, register_count, likelihood,
                                crew_hours, review_plan, requirements_analysis, evidence_review, prediction_analysis,
                                decision_rules):
    analyses = [f"{label}: {text}" for fields in (requirements_analysis, evidence_review, prediction_analysis)
                for label, text in fields.items()]
    labels = ["RECOMMENDATION", "RECOMMENDED DECISION"]
    fields, call_record = _ask([
        _speaker("Write the review a Highways Inspector decides from, and recommend one of the decisions open."),
        "The report:\n" + _report(report_profile),
        "What the person who opened the review asked: " + coordinator_request,
        f"The plan: PRIORITY: {review_plan['PRIORITY']}\nCHECKS: {review_plan['CHECKS']}",
        "The analysts' findings:\n" + "\n".join(analyses),
        "Each requirement:\n" + _requirements(screening),
        _permits(register_count) + "\n" + _screening(screening),
        f"Repeat likelihood {likelihood['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f} "
        f"({likelihood['likelihood_band']}). {describe_expected_crew_hours(crew_hours)}",
        "What the Highways Policy says about the three decisions:\n" + decision_rules,
        f"Recommend the assessment, {screening['assessment']}, unless the review justifies another decision open "
        f"and says why. Neither figure may be a reason. Present nothing as decided, rejected or closed. "
        f"RECOMMENDED DECISION is exactly one of: {_list(screening['decisions_open'])}."],
        labels, config.AGENT_NAME_WRITER)
    _one_line(fields, "RECOMMENDED DECISION", config.AGENT_NAME_WRITER)
    decision = re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", fields["RECOMMENDED DECISION"]).upper()
    if decision not in screening["decisions_open"]:
        raise ValueError(f"{config.AGENT_NAME_WRITER} recommended {fields['RECOMMENDED DECISION']}, which is not "
                         f"among the decisions open")
    return {"review_recommendation": {**fields, "RECOMMENDED DECISION": decision}, "call_record": call_record}


def _order_details(document_details):
    lines = [f"- {label}: {describe_record(document_details)[column]}"
             for column, label in config.DOCUMENT_DETAIL_LABELS.items()]
    for defect_type, measurements in document_details["measurements"].items():
        lines += [f"- {defect_type}, {_label(column)}: {value}"
                  for column, value in describe_record(measurements).items()]
    return "\n".join(lines)


def _inspection_details(document_details):
    lines = [f"- Report reference: {document_details[config.REPORT_REFERENCE_COLUMN]}",
             "- Standards screened: " + _list(config.DEFECT_STANDARD_DOCUMENTS[defect_type]
                                              for defect_type in document_details["defect_types"])]
    for item in document_details["requirements_to_settle"]:
        lines += [f"- To settle, {item['defect_type']} standard, {item['requirement_name']} ({item['result']}): "
                  f"{item['requirement_sentence']}",
                  _answers(item["report_answers"], list(item["report_answers"])),
                  "  Answers needed: " + _list(_label(column) for column in item["blank_answers"])]
    lines += [f"- What the {defect_type} standard says an inspection records: {text}"
              for defect_type, text in document_details["inspection_records"].items()]
    lines += [f"- Flagged sentence {flag['document_id']}: \"{flag['sentence']}\" {flag['flag']}"
              for flag in document_details["flagged_statements"]]
    return "\n".join(lines)


def draft_order_document(decision, report_profile, document_details, document_guidance):
    order = decision == config.DECISION_ORDER
    kind = config.DOCUMENT_KINDS[decision]
    labels = ["SUBJECT", "DOCUMENT"]
    fields, call_record = _ask([
        _speaker(f"Write the {kind} to {config.DOCUMENT_RECIPIENTS[decision]}."),
        f"The decision: {decision}, meaning {config.DECISION_DESCRIPTIONS[decision]}.",
        f"The report came in by {describe_record(report_profile).get('report_channel')}.",
        "The details:\n" + (_order_details(document_details) if order else _inspection_details(document_details)),
        "Guidance:\n" + document_guidance,
        f"The SUBJECT is one line naming the report reference. The DOCUMENT names every defect type and the street"
        f"{'' if order else ' and every requirement to settle'}, holds no model figure, no staff name, no square "
        f"brackets and no invented time, date or contact detail, never uses the words "
        f"{_list(config.DOCUMENT_FORBIDDEN_TERMS)}, and ends with: {config.DOCUMENT_SIGN_OFF}"],
        labels, config.AGENT_NAME_ORDER_WRITER)
    agent, reference = config.AGENT_NAME_ORDER_WRITER, document_details[config.REPORT_REFERENCE_COLUMN]
    _one_line(fields, "SUBJECT", agent)
    if reference.lower() not in fields["SUBJECT"].lower():
        raise ValueError(f"{agent} wrote a subject that does not name {reference}")
    text = _flat(fields["SUBJECT"] + " " + fields["DOCUMENT"])
    if order:
        wanted = list(document_details["measurements"]) + [document_details[config.STREET_NAME_COLUMN]]
    else:
        wanted = [item["requirement_name"] for item in document_details["requirements_to_settle"]]
    missing = [name for name in wanted if _flat(name) not in text]
    if missing:
        raise ValueError(f"{agent} wrote a {kind} that does not name {_list(missing)}")
    _refuse_wording(text, fields["DOCUMENT"], agent)
    return {"order_document": fields, "call_record": call_record}


def draft_resident_update(works_order_details, update_guidance):
    details = "\n".join(f"- {label}: {describe_record(works_order_details)[column]}"
                        for column, label in config.DOCUMENT_DETAIL_LABELS.items())
    labels = ["HEADLINE", "UPDATE"]
    fields, call_record = _ask([
        _speaker("Write the update for the resident who reported the defect, now that a works order has been sent."),
        "The details:\n" + details,
        "What each standard says the update tells the reporter:\n" + update_guidance,
        f"The HEADLINE is one line. Name the report reference, the street and any linked defect type. Never say the "
        f"defect was visited, inspected or measured, promise no time for the work, give no review result, no model "
        f"figure, no staff name, no square brackets and no invented time, date or contact detail, never use the "
        f"words {_list(config.DOCUMENT_FORBIDDEN_TERMS)}, and end the UPDATE with: {config.DOCUMENT_SIGN_OFF}"],
        labels, config.AGENT_NAME_UPDATE_WRITER)
    agent = config.AGENT_NAME_UPDATE_WRITER
    _one_line(fields, "HEADLINE", agent)
    text = _flat(fields["HEADLINE"] + " " + fields["UPDATE"])
    linked = works_order_details.get(config.LINKED_DEFECT_TYPE_COLUMN)
    wanted = [works_order_details[config.REPORT_REFERENCE_COLUMN], works_order_details[config.STREET_NAME_COLUMN]]
    wanted += [linked] if linked and linked != config.NO_LINKED_DEFECT else []
    missing = [name for name in wanted if _flat(name) not in text]
    staff = [account["username"] for account in config.SEED_USER_ACCOUNTS
             if re.search(config.NAME_IN_TEXT_PATTERN.format(name=re.escape(account["username"].lower())), text)
             or _flat(account["display_name"]) in text]
    if missing or staff:
        raise ValueError(f"{agent} wrote an update that misses {_list(missing)} or names {_list(staff)}")
    _refuse_wording(text, fields["UPDATE"], agent)
    return {"resident_update": fields, "call_record": call_record}
