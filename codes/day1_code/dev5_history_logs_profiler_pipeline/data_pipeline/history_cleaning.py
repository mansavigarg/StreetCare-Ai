from config import config
from data_pipeline.common_cleaning import (convert_readings, convert_report_references, convert_whole_numbers,
                                           keep_latest_revision_of_each_report, normalize_category_values,
                                           remove_duplicate_records, remove_rows_missing_target,
                                           treat_values_above_maximum_as_missing,
                                           treat_values_below_minimum_as_missing)


def read_history_answers(frame):
    history = frame.copy()
    history[config.REPORT_REFERENCE_COLUMN] = convert_report_references(history[config.REPORT_REFERENCE_COLUMN])
    history = normalize_category_values(history, config.HISTORY_VALUE_MAPPINGS)
    history = convert_whole_numbers(history, config.HISTORY_WHOLE_NUMBER_PATTERNS)
    return convert_readings(history, config.FORM_READINGS)


def clean_history_dataset(frame):
    history = read_history_answers(remove_duplicate_records(frame))
    history = treat_values_below_minimum_as_missing(history, config.FORM_MINIMUM_VALUES)
    history = treat_values_above_maximum_as_missing(history, config.FORM_MAXIMUM_VALUES)
    history = keep_latest_revision_of_each_report(history)
    return remove_rows_missing_target(history, config.HISTORY_TARGET_COLUMN)
