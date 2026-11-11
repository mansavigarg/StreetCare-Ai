from config import config
from data_pipeline.common_cleaning import (convert_markup_to_plain_text, convert_report_references,
                                           normalize_category_values, refuse_missing_values,
                                           refuse_repeated_references, refuse_unreadable_identifiers,
                                           remove_duplicate_records)


def clean_document_records(frame):
    documents = remove_duplicate_records(frame)
    refuse_missing_values(documents, [config.DOCUMENT_ID_COLUMN, config.REPORT_REFERENCE_COLUMN])
    refuse_unreadable_identifiers(documents, config.DOCUMENT_IDENTIFIER_PATTERNS)
    refuse_repeated_references(documents, config.DOCUMENT_ID_COLUMN)
    documents = normalize_category_values(documents, config.DOCUMENT_VALUE_MAPPINGS)
    documents[config.REPORT_REFERENCE_COLUMN] = convert_report_references(documents[config.REPORT_REFERENCE_COLUMN])
    documents[config.DOCUMENT_TEXT_COLUMN] = convert_markup_to_plain_text(documents[config.DOCUMENT_TEXT_COLUMN])
    return documents
