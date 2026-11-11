import re
from decimal import Decimal

import pandas as pd

from config import config


def _blank(value):
    return value is None or pd.isna(value)


def read_document_evidence(sentence_model, documents):
    pieces = []
    for document in documents:
        text = document[config.DOCUMENT_TEXT_COLUMN]
        if _blank(text) or not str(text).strip():
            raise ValueError(config.NO_STATEMENT_TEXT_MESSAGE.format(document_id=document[config.DOCUMENT_ID_COLUMN]))
        for sentence in re.split(config.STATEMENT_SENTENCE_PATTERN, text):
            pieces.append((document, sentence))
    if not pieces:
        return []
    types = sentence_model.predict([sentence for _, sentence in pieces])
    return [{"document_id": document[config.DOCUMENT_ID_COLUMN], "document_type": document[config.DOCUMENT_TYPE_COLUMN],
             "sentence": sentence, "sentence_type": str(sentence_type)}
            for (document, sentence), sentence_type in zip(pieces, types)]


def link_evidence_to_requirements(findings, screening):
    linked = []
    for finding in findings:
        sentence_type = finding["sentence_type"]
        if sentence_type not in config.SENTENCE_TYPE_REQUIREMENT_KINDS:
            raise ValueError(config.NO_REQUIREMENT_KINDS_MESSAGE.format(sentence_type=sentence_type))
        kinds = config.SENTENCE_TYPE_REQUIREMENT_KINDS[sentence_type]
        borne = [{"defect_type": item["defect_type"], "requirement_name": item["requirement_name"]}
                 for item in screening["requirements"] if item["requirement_kind"] in kinds]
        linked.append({**finding, "requirements_borne_on": borne})
    return linked


def find_disagreements(findings, report):
    flags = []
    for finding in findings:
        if finding["sentence_type"] != config.MEASUREMENT:
            continue
        stated = {column: re.findall(pattern, finding["sentence"])
                  for column, pattern in config.STATEMENT_MEASUREMENT_PATTERNS.items()}
        stated = {column: figures for column, figures in stated.items() if figures}
        if not stated:
            continue
        on_report = {column: None if _blank(report.get(column)) else report.get(column) for column in stated}
        if any(value is None for value in on_report.values()):
            flag = config.DESCRIBES_BLANK_ANSWER
        elif any(Decimal(figure) != Decimal(str(on_report[column])) for column, figures in stated.items()
                 for figure in figures):
            flag = config.DISAGREES_WITH_FORM
        else:
            continue
        flags.append({"document_id": finding["document_id"], "sentence": finding["sentence"],
                      "stated_readings": {column: [float(figure) for figure in figures]
                                          for column, figures in stated.items()},
                      "report_readings": {column: None if value is None else float(value)
                                          for column, value in on_report.items()},
                      "flag": flag})
    return flags
