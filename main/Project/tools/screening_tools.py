from decimal import Decimal

import pandas as pd

from config import config


def _blank(value):
    return value is None or pd.isna(value)


def _exact(value):
    return Decimal(str(value))


def _defect_types(report):
    linked = report.get(config.LINKED_DEFECT_TYPE_COLUMN)
    own = [report[config.DEFECT_TYPE_COLUMN]]
    return own if _blank(linked) or linked == config.NO_LINKED_DEFECT else own + [linked]


def count_live_street_works(register, report):
    street = report.get(config.STREET_NAME_COLUMN)
    if _blank(street):
        raise ValueError(config.BLANK_STREET_MESSAGE.format(report_reference=report[config.REPORT_REFERENCE_COLUMN]))
    live = register[(register[config.STREET_NAME_COLUMN] == street)
                    & (register[config.PERMIT_STATUS_COLUMN] == config.PERMIT_STATUS_LIVE)]
    live = live.sort_values(config.PERMIT_REFERENCE_COLUMN)
    parts = {config.DEFECT_TYPE_HIGHWAY_PARTS[kind] for kind in _defect_types(report)
             if kind in config.DEFECT_TYPE_HIGHWAY_PARTS}
    conflicting = live[live[config.WORKS_LOCATION_COLUMN].isin(parts)]
    live_permits = list(live[config.PERMIT_REFERENCE_COLUMN])
    conflicting_permits = list(conflicting[config.PERMIT_REFERENCE_COLUMN])
    return {config.LIVE_STREET_WORKS_COLUMN: len(live_permits),
            config.CONFLICTING_STREET_WORKS_COLUMN: len(conflicting_permits),
            "live_permits": live_permits, "conflicting_permits": conflicting_permits}


def find_coordination_triggers(report):
    conflicts = report.get(config.CONFLICTING_STREET_WORKS_COLUMN)
    if _blank(conflicts):
        raise ValueError(config.BLANK_STREET_WORKS_COUNT_MESSAGE.format(
            report_reference=report[config.REPORT_REFERENCE_COLUMN]))
    kinds = _defect_types(report)
    category, height = report.get(config.STREET_CATEGORY_COLUMN), report.get(config.COLUMN_HEIGHT_COLUMN)
    triggers = []
    if conflicts > 0:
        triggers.append(config.LIVE_STREET_WORKS_TRIGGER)
    if set(kinds) & set(config.TRAFFIC_SENSITIVE_DEFECT_TYPES) and (_blank(category) or category != config.LOCAL_ROAD):
        triggers.append(config.TRAFFIC_SENSITIVE_TRIGGER)
    if config.STREET_LIGHTING in kinds and (_blank(height) or _exact(height) > config.HIGH_COLUMN_TALLER_THAN_METRES):
        triggers.append(config.HIGH_COLUMN_TRIGGER)
    return triggers


def _within(value, rules):
    value = _exact(value)
    return (value >= rules.get(config.MINIMUM_RULE, value)) and (value <= rules.get(config.MAXIMUM_RULE, value))


def _judge(report, kind, rules):
    columns = config.REQUIREMENT_ANSWER_COLUMNS[kind]
    first = report.get(columns[0])
    if _blank(first):
        return config.REQUIREMENT_NOT_DOCUMENTED, [columns[0]]
    if len(columns) == 1:
        if first in (config.BOOLEAN_YES, config.BOOLEAN_NO):
            return (config.REQUIREMENT_MET if first == config.BOOLEAN_YES else config.REQUIREMENT_NOT_MET), []
        return (config.REQUIREMENT_MET if _within(first, rules) else config.REQUIREMENT_NOT_MET), []
    follow_on = report.get(columns[1])
    if kind == config.WATER_REACH_REQUIREMENT and first == config.FOOTWAY_PART:
        return config.REQUIREMENT_MET, []
    if kind == config.COLUMN_LEAN_REQUIREMENT and first != config.LEANING_COLUMN:
        return config.REQUIREMENT_MET, []
    if _blank(follow_on):
        return config.REQUIREMENT_NOT_DOCUMENTED, [columns[1]]
    if kind == config.CLEAR_HEIGHT_REQUIREMENT:
        limit = rules[config.FOOTWAY_LIMIT_RULE if first == config.FOOTWAY_PART else config.CARRIAGEWAY_LIMIT_RULE]
        return (config.REQUIREMENT_MET if _exact(follow_on) < limit else config.REQUIREMENT_NOT_MET), []
    return (config.REQUIREMENT_MET if _within(follow_on, rules) else config.REQUIREMENT_NOT_MET), []


def screen_report(report, standards):
    reference = report[config.REPORT_REFERENCE_COLUMN]
    defect_types = _defect_types(report)
    requirements = []
    for defect_type in defect_types:
        standard = standards.get(defect_type)
        if standard is None:
            raise ValueError(config.NO_STANDARD_READ_MESSAGE.format(report_reference=reference, defect_type=defect_type))
        if standard["defect_type"] != defect_type:
            raise ValueError(config.STANDARD_OF_ANOTHER_DEFECT_TYPE_MESSAGE.format(
                defect_type=defect_type, document_name=standard["document_name"],
                stated_defect_type=standard["defect_type"]))
        for requirement in standard["requirements"]:
            kind = requirement["requirement_kind"]
            if kind not in config.REQUIREMENT_ANSWER_COLUMNS:
                raise ValueError(f"{standard['document_name']} states the requirement kind {kind}, "
                                 f"which the screening does not know")
            result, blanks = _judge(report, kind, requirement["rules"])
            requirements.append({"defect_type": defect_type, "standard_document": standard["document_name"],
                                 "requirement_kind": kind, "requirement_name": requirement["requirement_name"],
                                 "requirement_sentence": requirement["requirement_sentence"],
                                 "rules": requirement["rules"], "result": result, "blank_answers": blanks})
    counts = {column: sum(item["result"] == result for item in requirements)
              for column, result in config.REQUIREMENT_COUNT_RESULTS.items()}
    triggers = find_coordination_triggers(report)
    if counts[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] or counts[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN]:
        assessment = config.DECISION_INSPECT
    else:
        assessment = config.DECISION_REFER if triggers else config.DECISION_ORDER
    return {"report_reference": reference, "defect_types": defect_types, "requirements": requirements, **counts,
            "coordination_triggers": triggers, "assessment": assessment,
            "decisions_open": config.DECISIONS_OPEN[assessment]}
