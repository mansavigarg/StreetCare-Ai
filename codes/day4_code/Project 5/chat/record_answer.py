from agentic.language_model import parse_labelled_response
from config import config
from persistence.models import ReportReview
from persistence.repository import fetch_review_records
from tools.report_data_tools import describe_record, fetch_report


def _joined(items, separator=", ", empty=config.CHAT_NONE_TEXT):
    return separator.join(items) or empty


def _team_lines(review, defect_types):
    standards = {}
    for item in review["requirements"]:
        standards.setdefault(item["defect_type"], []).append(f"{item['requirement_name']} {item['result']}")
    requirements = _joined([f"{kind}: {', '.join(results)}" for kind, results in standards.items()], "; ")
    permits = [config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=permit)
               if permit in review["conflicting_permits"] else permit for permit in review["live_permits"]]
    sentence_types = list(dict.fromkeys(finding["sentence_type"] for finding in review["evidence"]))
    if review["status"] == config.REVIEW_STATUS_AWAITING:
        last = config.CHAT_AWAITING_DECISION_TEMPLATE.format(recommended_decision=review["recommended_decision"])
    else:
        last = config.CHAT_DECIDED_TEMPLATE.format(**{key: review[key] for key in (
            "decided_by", "decided_by_role", "decided_by_district", "decision", "decision_reason")})
    return [config.CHAT_REVIEW_TEMPLATE.format(defect_types=defect_types,
                                               status=config.REPORT_STATUS_PHRASES[review["status"]],
                                               **{key: review[key] for key in (
                                                   "review_id", "report_reference", "revision_number",
                                                   "street_name", "coordinator_id")}),
            config.CHAT_SCREENING_TEMPLATE.format(assessment=review["assessment"], requirements=requirements,
                                                  triggers=_joined(review["coordination_triggers"], "; "),
                                                  decisions_open=_joined(review["decisions_open"])),
            config.CHAT_PERMITS_TEMPLATE.format(live_permits=_joined(permits)),
            config.CHAT_FIGURES_TEMPLATE.format(
                repeat_likelihood=f"{review['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}",
                likelihood_band=review["likelihood_band"], crew_hours_text=review["crew_hours_text"]),
            config.CHAT_EVIDENCE_TEMPLATE.format(
                sentence_count=len(review["evidence"]), flag_count=len(review["disagreements"]),
                sentence_types=_joined(sentence_types, empty=config.CHAT_NO_DOCUMENTS_TEXT)),
            last]


def answer_report_record(session, signed_in_user, open_reports, question_type, report_reference):
    adviser = signed_in_user["role"] == config.ROLE_CUSTOMER_SERVICES_ADVISER
    if adviser and question_type == config.CHAT_TYPE_OUTCOME_PREDICTION:
        return config.OUTCOME_NOT_PREDICTED_LINE
    if report_reference is None:
        return config.CHAT_REPORT_NOT_NAMED_MESSAGE
    if report_reference not in set(open_reports[config.REPORT_REFERENCE_COLUMN]):
        return config.CHAT_NO_REPORT_TEMPLATE.format(report_reference=report_reference)
    report = fetch_report(open_reports, report_reference)
    described = describe_record(report)
    defect_types = described[config.DEFECT_TYPE_COLUMN]
    if report[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT:
        defect_types = config.CHAT_LINKED_DEFECT_TEMPLATE.format(
            defect_type=defect_types, linked_defect_type=described[config.LINKED_DEFECT_TYPE_COLUMN])
    reviews = fetch_review_records(session, (ReportReview.report_reference == report_reference)
                                   & (ReportReview.revision_number == report[config.REVISION_NUMBER_COLUMN]))
    review = reviews[-1] if reviews else None
    if adviser:
        status = config.REPORT_STATUS_AWAITING_REVIEW if review is None else review["status"]
        answer = config.CHAT_ADVISER_STATUS_TEMPLATE.format(
            report_reference=report_reference, defect_types=defect_types,
            street_name=described[config.STREET_NAME_COLUMN], status=config.REPORT_STATUS_PHRASES[status])
        if status == config.REVIEW_STATUS_REPAIR_ORDERED:
            update = parse_labelled_response(review["resident_update_text"], ["HEADLINE", "UPDATE"],
                                             config.AGENT_NAME_UPDATE_WRITER)["UPDATE"]
            answer += " " + config.CHAT_ORDERED_REPAIR_TEMPLATE.format(
                location_description=describe_record(review["document_details"])[config.LOCATION_DESCRIPTION_COLUMN],
                resident_update=" ".join(update.split()))
        return answer
    if review is None:
        return config.CHAT_NO_REVIEW_TEMPLATE.format(report_reference=report_reference)
    return "\n".join(_team_lines(review, defect_types))


def answer_street_figure(report_history, street_name, defect_type):
    if street_name is None or defect_type is None:
        return config.CHAT_STREET_NOT_NAMED_MESSAGE
    rows = report_history[(report_history[config.STREET_NAME_COLUMN] == street_name)
                          & (report_history[config.DEFECT_TYPE_COLUMN] == defect_type)]
    count = rows[config.REPORT_REFERENCE_COLUMN].nunique()
    if count < config.GROUP_SIZE_MINIMUM:
        return config.GROUP_SIZE_LINE
    return config.CHAT_STREET_FIGURE_TEMPLATE.format(report_count=count, defect_type=defect_type.lower(),
                                                     street_name=street_name)
