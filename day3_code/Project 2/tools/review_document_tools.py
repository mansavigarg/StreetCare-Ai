import pandas as pd

from config import config


def _plain(value):
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _same_report(report, screening):
    if screening["report_reference"] != report[config.REPORT_REFERENCE_COLUMN]:
        raise ValueError(config.SCREENING_OF_ANOTHER_REPORT_MESSAGE.format(
            screening_reference=screening["report_reference"], report_reference=report[config.REPORT_REFERENCE_COLUMN]))


def _details(report, columns):
    return {config.REPORT_REFERENCE_COLUMN: report[config.REPORT_REFERENCE_COLUMN],
            **{column: _plain(report.get(column)) for column in columns}}


def _flagged_statements(disagreements):
    return [{key: flag[key] for key in ("document_id", "sentence", "flag")} for flag in disagreements]


def build_works_order_details(report, screening):
    _same_report(report, screening)
    measurements = {defect_type: {column: _plain(report.get(column))
                                  for column in config.WORKS_ORDER_MEASUREMENT_COLUMNS[defect_type]
                                  if _plain(report.get(column)) is not None}
                    for defect_type in screening["defect_types"]}
    return {**_details(report, config.WORKS_ORDER_DETAIL_COLUMNS), "measurements": measurements}


def build_inspection_request_details(report, screening, inspection_records, disagreements):
    _same_report(report, screening)
    if list(inspection_records) != list(screening["defect_types"]):
        raise ValueError(f"The inspection records {list(inspection_records)} are not those of the screened "
                         f"defect types {list(screening['defect_types'])}")
    to_settle = [{"defect_type": item["defect_type"], "requirement_name": item["requirement_name"],
                  "requirement_sentence": item["requirement_sentence"], "result": item["result"],
                  "report_answers": {column: _plain(report.get(column))
                                     for column in config.REQUIREMENT_ANSWER_COLUMNS[item["requirement_kind"]]},
                  "blank_answers": list(item["blank_answers"])}
                 for item in screening["requirements"] if item["result"] != config.REQUIREMENT_MET]
    return {config.REPORT_REFERENCE_COLUMN: report[config.REPORT_REFERENCE_COLUMN],
            "defect_types": list(screening["defect_types"]), "requirements_to_settle": to_settle,
            "inspection_records": dict(inspection_records), "flagged_statements": _flagged_statements(disagreements)}


def build_coordination_brief(report, screening, register_count, likelihood, crew_hours, disagreements):
    _same_report(report, screening)
    requirements = [{key: item[key] for key in ("defect_type", "requirement_name", "result", "blank_answers")}
                    for item in screening["requirements"]]
    return {**_details(report, config.COORDINATION_BRIEF_DETAIL_COLUMNS), "requirements": requirements,
            "coordination_triggers": list(screening["coordination_triggers"]),
            "live_permits": list(register_count["live_permits"]),
            "conflicting_permits": list(register_count["conflicting_permits"]),
            "flagged_statements": _flagged_statements(disagreements),
            "repeat_likelihood": likelihood["repeat_likelihood"], "likelihood_band": likelihood["likelihood_band"],
            **{key: crew_hours[key] for key in ("expected_crew_hours", "planned_crew_hours", "overrun_expected")}}
