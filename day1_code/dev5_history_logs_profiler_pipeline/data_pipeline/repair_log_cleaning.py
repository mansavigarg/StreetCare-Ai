from config import config
from data_pipeline.common_cleaning import (convert_readings, convert_report_references, convert_whole_numbers,
                                           normalize_category_values, refuse_missing_values,
                                           refuse_repeated_references, remove_duplicate_records,
                                           remove_rows_missing_target, treat_values_above_maximum_as_missing,
                                           treat_values_below_minimum_as_missing)


def read_log_answers(frame):
    logs = frame.copy()
    logs[config.REPORT_REFERENCE_COLUMN] = convert_report_references(logs[config.REPORT_REFERENCE_COLUMN])
    logs = normalize_category_values(logs, config.LOG_VALUE_MAPPINGS)
    logs = convert_whole_numbers(logs, config.LOG_WHOLE_NUMBER_PATTERNS)
    return convert_readings(logs, config.LOG_READINGS)


def clean_repair_log_dataset(frame):
    logs = refuse_missing_values(remove_duplicate_records(frame), [config.REPORT_REFERENCE_COLUMN])
    logs = refuse_repeated_references(read_log_answers(logs), config.REPORT_REFERENCE_COLUMN)
    logs = treat_values_below_minimum_as_missing(logs, config.LOG_MINIMUM_VALUES)
    logs = treat_values_above_maximum_as_missing(logs, config.LOG_MAXIMUM_VALUES)
    return remove_rows_missing_target(logs, config.LOG_TARGET_COLUMN)
