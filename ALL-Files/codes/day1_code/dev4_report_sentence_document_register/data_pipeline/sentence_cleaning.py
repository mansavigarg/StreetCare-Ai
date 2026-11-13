from config import config
from data_pipeline.common_cleaning import (convert_markup_to_plain_text, convert_report_references,
                                           convert_whole_numbers, lower_text_written_in_capitals,
                                           normalize_category_values, refuse_missing_values,
                                           refuse_repeated_references, refuse_unreadable_identifiers,
                                           remove_duplicate_records, remove_rows_missing_target)


def clean_sentence_dataset(frame):
    sentences = remove_duplicate_records(frame)
    refuse_missing_values(sentences, [config.SENTENCE_ID_COLUMN, config.REPORT_REFERENCE_COLUMN])
    refuse_unreadable_identifiers(sentences, config.SENTENCE_IDENTIFIER_PATTERNS)
    refuse_repeated_references(sentences, config.SENTENCE_ID_COLUMN)
    sentences = normalize_category_values(sentences, config.SENTENCE_VALUE_MAPPINGS)
    sentences[config.REPORT_REFERENCE_COLUMN] = convert_report_references(sentences[config.REPORT_REFERENCE_COLUMN])
    sentences = convert_whole_numbers(sentences, config.SENTENCE_WHOLE_NUMBER_PATTERNS)
    text = lower_text_written_in_capitals(convert_markup_to_plain_text(sentences[config.SENTENCE_TEXT_COLUMN]))
    sentences[config.SENTENCE_TEXT_COLUMN] = text
    sentences[config.SENTENCE_LENGTH_COLUMN] = text.str.len().astype(config.WHOLE_NUMBER_DTYPE)
    sentences = remove_rows_missing_target(sentences, config.SENTENCE_TARGET_COLUMN)
    words = sentences[config.SENTENCE_TEXT_COLUMN].str.split().str.len().fillna(0)
    return sentences[words >= config.MINIMUM_SENTENCE_WORDS].reset_index(drop=True)
