import json
import os
import re

from config import config
from data_pipeline.common_cleaning import (convert_markup_to_plain_text, remove_duplicate_records,
                                           treat_values_above_maximum_as_missing,
                                           treat_values_below_minimum_as_missing)
from data_pipeline.history_cleaning import read_history_answers
from data_pipeline.repair_log_cleaning import read_log_answers


def _gaps(frame):
    counts = frame.isna().sum()
    return {column: int(count) for column, count in counts.items() if count}


def _canonical_reference_misses(frame):
    references = frame[config.REPORT_REFERENCE_COLUMN].dropna()
    return int((~references.str.fullmatch(config.CANONICAL_REPORT_REFERENCE_PATTERN)).sum())


def _non_canonical_categories(frame, value_mappings):
    counts = {}
    for column, mapping in value_mappings.items():
        values = frame[column].dropna()
        misses = int((~values.isin(set(mapping.values()))).sum())
        if misses:
            counts[column] = misses
    return counts


def _readings_by_unit(frame, readings):
    counts = {}
    for column, reading in readings.items():
        values = frame[column].dropna()
        counts[column] = {notation: int(values.str.fullmatch(pattern).sum())
                          for notation, pattern in reading["notations"].items()}
    return counts


def _outside_limits(read_frame, limits, treat):
    changed = treat(read_frame, limits)
    counts = {column: int((read_frame[column].notna() & changed[column].isna()).sum()) for column in limits}
    return {column: count for column, count in counts.items() if count}


def _target_shares(clean_frame, target_column):
    shares = clean_frame[target_column].value_counts(normalize=True)
    return {str(answer): round(float(share), config.TARGET_SHARE_DECIMALS) for answer, share in shares.items()}


def profile_dataset(raw_frame, clean_frame, target_column):
    received = remove_duplicate_records(raw_frame)
    return {"rows_received": len(raw_frame), "rows_kept": len(clean_frame),
            "rows_removed": len(raw_frame) - len(clean_frame), "duplicate_rows": len(raw_frame) - len(received),
            "rows_missing_target": int(received[target_column].isna().sum()), "columns": len(clean_frame.columns),
            "missing_cells_as_received": _gaps(received), "missing_cells_after_cleaning": _gaps(clean_frame)}


def _removed_beyond_target(raw_frame, clean_frame, target_column):
    received = remove_duplicate_records(raw_frame)
    return len(received) - len(clean_frame) - int(received[target_column].isna().sum())


def profile_history_details(raw_frame, clean_frame):
    received = remove_duplicate_records(raw_frame)
    read = read_history_answers(received)
    return {"answered_earlier_revisions_removed": _removed_beyond_target(raw_frame, clean_frame, config.HISTORY_TARGET_COLUMN),
            "report_references_not_written_canonically": _canonical_reference_misses(received),
            "readings_by_unit_as_written": _readings_by_unit(received, config.FORM_READINGS),
            "values_below_minimum_as_received": _outside_limits(read, config.FORM_MINIMUM_VALUES, treat_values_below_minimum_as_missing),
            "values_above_maximum_as_received": _outside_limits(read, config.FORM_MAXIMUM_VALUES, treat_values_above_maximum_as_missing),
            "category_values_not_written_canonically": _non_canonical_categories(received, config.HISTORY_VALUE_MAPPINGS),
            "target_shares": _target_shares(clean_frame, config.HISTORY_TARGET_COLUMN)}


def profile_log_details(raw_frame, clean_frame):
    received = remove_duplicate_records(raw_frame)
    read = read_log_answers(received)
    return {"rows_with_impossible_target": _removed_beyond_target(raw_frame, clean_frame, config.LOG_TARGET_COLUMN),
            "report_references_not_written_canonically": _canonical_reference_misses(received),
            "readings_by_unit_as_written": _readings_by_unit(received, config.LOG_READINGS),
            "values_below_minimum_as_received": _outside_limits(read, config.LOG_MINIMUM_VALUES, treat_values_below_minimum_as_missing),
            "values_above_maximum_as_received": _outside_limits(read, config.LOG_MAXIMUM_VALUES, treat_values_above_maximum_as_missing),
            "category_values_not_written_canonically": _non_canonical_categories(received, config.LOG_VALUE_MAPPINGS),
            "median_actual_crew_hours": float(clean_frame[config.LOG_TARGET_COLUMN].median())}


def profile_sentence_details(raw_frame, clean_frame):
    received = remove_duplicate_records(raw_frame)
    texts = received[config.SENTENCE_TEXT_COLUMN].fillna("")
    plain = convert_markup_to_plain_text(received[config.SENTENCE_TEXT_COLUMN]).fillna("")
    lengths = received[config.SENTENCE_LENGTH_COLUMN].fillna("")
    return {"sentences_with_markup": int(texts.str.contains(config.MARKUP_PATTERN).sum()),
            "sentences_with_encoding_artifacts": int(texts.str.contains(config.ENCODING_ARTIFACT_PATTERN).sum()),
            "sentences_in_capitals": int(plain.str.isupper().sum()),
            "sentence_lengths_disagreeing_with_plain_text_as_received": int((lengths != plain.str.len().astype(str)).sum()),
            "rows_with_too_few_words": _removed_beyond_target(raw_frame, clean_frame, config.SENTENCE_TARGET_COLUMN),
            "report_references_not_written_canonically": _canonical_reference_misses(received),
            "shortest_sentence_words": int(clean_frame[config.SENTENCE_TEXT_COLUMN].str.split().str.len().min()),
            "category_values_not_written_canonically": _non_canonical_categories(received, config.SENTENCE_VALUE_MAPPINGS),
            "target_shares": _target_shares(clean_frame, config.SENTENCE_TARGET_COLUMN)}


def write_quality_report(profiles):
    os.makedirs(os.path.dirname(config.DATA_QUALITY_REPORT_FILE), exist_ok=True)
    with open(config.DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as report:
        json.dump(profiles, report, indent=config.JSON_INDENT, sort_keys=True)
