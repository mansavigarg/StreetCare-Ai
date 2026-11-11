from config import config
from data_pipeline.common_cleaning import (normalize_category_values, refuse_missing_values,
                                           refuse_repeated_references, refuse_unreadable_identifiers,
                                           remove_duplicate_records)


def clean_register_records(frame):
    permits = remove_duplicate_records(frame)
    refuse_missing_values(permits, config.REGISTER_REQUIRED_COLUMNS)
    refuse_unreadable_identifiers(permits, config.REGISTER_IDENTIFIER_PATTERNS)
    refuse_repeated_references(permits, config.PERMIT_REFERENCE_COLUMN)
    return normalize_category_values(permits, config.REGISTER_VALUE_MAPPINGS)
