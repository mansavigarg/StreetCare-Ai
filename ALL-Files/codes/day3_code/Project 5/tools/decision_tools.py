from config import config


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def find_review_conflict(review_status, pending_decision, review_decision, report_reference):
    if review_status in config.DECIDED_REVIEW_STATUSES:
        return config.ALREADY_DECIDED_MESSAGE.format(report_reference=report_reference)
    if pending_decision is not None and pending_decision != review_decision:
        return config.DECISION_IN_PROGRESS_MESSAGE.format(pending_decided_by=pending_decision["decided_by"],
                                                          pending_decision=pending_decision["decision"],
                                                          report_reference=report_reference)
    return None


def find_decision_refusal(review_decision, report, screening, opened_by):
    reference = report[config.REPORT_REFERENCE_COLUMN]
    if screening["report_reference"] != reference:
        raise ValueError(config.SCREENING_OF_ANOTHER_REPORT_MESSAGE.format(
            screening_reference=screening["report_reference"], report_reference=reference))
    missing = [field for field in config.DECISION_RECORD_FIELDS if _blank(review_decision.get(field))]
    if missing:
        return config.DECISION_FIELDS_MISSING_MESSAGE.format(report_reference=reference, fields=", ".join(missing))
    decision, decided_by = review_decision["decision"], review_decision["decided_by"]
    if decision not in config.DECISIONS:
        return config.UNKNOWN_DECISION_MESSAGE.format(decision=decision, report_reference=reference,
                                                      decisions=", ".join(config.DECISIONS))
    if review_decision["decided_by_role"] != config.ROLE_HIGHWAYS_INSPECTOR:
        return config.DECIDING_ROLE_MESSAGE.format(role=review_decision["decided_by_role"], report_reference=reference,
                                                   inspector_role=config.ROLE_HIGHWAYS_INSPECTOR)
    district, street = review_decision["decided_by_district"], report[config.STREET_NAME_COLUMN]
    if street not in config.INSPECTION_DISTRICT_STREETS.get(district, []):
        return config.NOT_APPOINTED_MESSAGE.format(decided_by=decided_by, district=district, street_name=street,
                                                   report_reference=reference)
    if decided_by == opened_by:
        return config.OWN_REVIEW_MESSAGE.format(decided_by=decided_by, report_reference=reference)
    if decision not in screening["decisions_open"]:
        return config.DECISION_NOT_OPEN_MESSAGE.format(decision=decision, report_reference=reference,
                                                       reason=config.ORDER_CLOSED_REASONS[screening["assessment"]])
    return None
