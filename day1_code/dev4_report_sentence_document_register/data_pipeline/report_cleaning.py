from config import config
from data_pipeline.common_cleaning import (convert_readings, convert_report_references, convert_whole_numbers,
                                           keep_latest_revision_of_each_report, normalize_category_values,
                                           remove_duplicate_records, treat_values_above_maximum_as_missing,
                                           treat_values_below_minimum_as_missing)


def clean_report_records(frame):
    reports = remove_duplicate_records(frame)
    reports[config.REPORT_REFERENCE_COLUMN] = convert_report_references(reports[config.REPORT_REFERENCE_COLUMN])
    reports = normalize_category_values(reports, config.FORM_VALUE_MAPPINGS)
    reports = convert_whole_numbers(reports, config.LIVE_WHOLE_NUMBER_PATTERNS)
    reports = convert_readings(reports, config.FORM_READINGS)
    reports = treat_values_below_minimum_as_missing(reports, config.FORM_MINIMUM_VALUES)
    reports = treat_values_above_maximum_as_missing(reports, config.FORM_MAXIMUM_VALUES)
    return keep_latest_revision_of_each_report(reports)
