import numbers

import pandas as pd

from config import config
from data_pipeline.dataset_loader import load_processed_dataset, load_raw_dataset
from data_pipeline.document_cleaning import clean_document_records
from data_pipeline.register_cleaning import clean_register_records
from data_pipeline.report_cleaning import clean_report_records


def _plain(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _records(table):
    return [{column: _plain(value) for column, value in row.items()} for row in table.to_dict("records")]


def load_open_reports():
    return clean_report_records(load_raw_dataset(config.OPEN_DEFECT_REPORTS_FILE))


def load_report_documents():
    return clean_document_records(load_raw_dataset(config.REPORT_DOCUMENTS_FILE))


def load_street_works_register():
    return clean_register_records(load_raw_dataset(config.STREET_WORKS_REGISTER_FILE))


def load_report_history():
    tables = [load_processed_dataset(config.CLEAN_REPORT_HISTORY_TRAIN_FILE),
              load_processed_dataset(config.CLEAN_REPORT_HISTORY_EVAL_FILE)]
    return pd.concat(tables, ignore_index=True)


def fetch_report(reports, report_reference):
    rows = reports[reports[config.REPORT_REFERENCE_COLUMN] == report_reference]
    if rows.empty:
        raise ValueError(config.UNKNOWN_REPORT_MESSAGE.format(report_reference=report_reference))
    return _records(rows.head(1))[0]


def fetch_report_documents(documents, report):
    own = documents[documents[config.REPORT_REFERENCE_COLUMN] == report[config.REPORT_REFERENCE_COLUMN]]
    return _records(own.sort_values(config.DOCUMENT_ID_COLUMN))


def rank_report_queue(reports):
    first_type = config.QUEUE_FIRST_DEFECT_TYPE
    first = ((reports[config.DEFECT_TYPE_COLUMN] == first_type)
             | (reports[config.LINKED_DEFECT_TYPE_COLUMN] == first_type))
    return (sorted(reports[first][config.REPORT_REFERENCE_COLUMN])
            + sorted(reports[~first][config.REPORT_REFERENCE_COLUMN]))


def _describe_value(value):
    value = _plain(value)
    if value is None:
        return config.NOT_GIVEN_TEXT
    if isinstance(value, numbers.Number) and not isinstance(value, bool) and float(value).is_integer():
        return f"{int(value):,}"
    return str(value)


def describe_record(record):
    return {key: _describe_value(value) for key, value in record.items()}
