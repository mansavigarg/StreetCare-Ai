import re

import pandas as pd

from agentic.language_model import call_language_model, parse_labelled_response
from config import config
from data_pipeline.common_cleaning import convert_report_references
from knowledge.standard_reading import read_defect_standard
from knowledge.vector_index import search_documents
from tools.retrieval_tools import read_inspection_records, read_reviewer_checks

LABELS = ["QUESTION TYPE", "REPORT", "DEFECT TYPE", "ANSWER"]
STREETS = [street for streets in config.INSPECTION_DISTRICT_STREETS.values() for street in streets]


def _value(fields, label):
    return re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", fields[label]).strip()


def _matching(value, options):
    return next((option for option in options if option.lower() == value.lower()), None)


def _named(reference, text):
    number = int(reference[len(config.REPORT_REFERENCE_PREFIX):])
    return (reference.lower() in text.lower()
            or re.search(config.NUMBER_IN_TEXT_PATTERN.format(number=number), text) is not None)


def _knowledge(collection, document_entries):
    lines = []
    for defect_type in config.DEFECT_STANDARD_DOCUMENTS:
        standard = read_defect_standard(defect_type, document_entries)
        lines += [f"{defect_type} standard ({standard['document_name']}), {item['requirement_name']}: "
                  f"{item['requirement_sentence']}" for item in standard["requirements"]]
        lines.append(f"{defect_type} standard, what an inspection records: "
                     f"{read_inspection_records(collection, defect_type)}")
    lines += [f"{sentence_type} guide ({document_name}), reviewer's checks: "
              f"{read_reviewer_checks(collection, sentence_type)}"
              for sentence_type, document_name in config.SENTENCE_TYPE_GUIDE_DOCUMENTS.items()]
    return "\n".join(lines)


def answer_from_documents(collection, document_entries, question_text, conversation):
    found = search_documents(collection, [question_text, conversation["search_text"]])
    passages = {}
    for passage in sorted(found[0] + found[1], key=lambda item: item["distance"]):
        passages.setdefault(passage["chunk_id"], passage)
    passages = list(passages.values())
    prompt = "\n\n".join([
        f"You answer questions for the {config.OFFICE_NAME} of {config.ORGANISATION_NAME} from its street care "
        f"documents and from nothing else. Use only what is given below, and answer each requirement from its own "
        f"standard.",
        "Passages found:\n" + "\n".join(f"- ({passage['document_name']}) {passage['chunk_text']}"
                                         for passage in passages),
        "The standards and guides:\n" + _knowledge(collection, document_entries),
        "The conversation so far:\n" + conversation["history_text"],
        f"The street the last exchange was about: {conversation['previous_street'] or config.CHAT_NO_VALUE_REPLY}",
        "The question: " + question_text,
        f"QUESTION TYPE is exactly one of: {', '.join(config.CHAT_TYPE_OPTIONS)}. A question about one report is "
        f"{config.CHAT_TYPE_REPORT_STATUS}, or {config.CHAT_TYPE_OUTCOME_PREDICTION} when it asks whether it will "
        f"come back or how long its repair will take; a question about how many defects a street has is "
        f"{config.CHAT_TYPE_STREET_FIGURE}. Those are answered from the records, never by you. A question that "
        f"leaves out its subject carries on from the last exchange.",
        f"REPORT is the report reference the question is about, or for a {config.CHAT_TYPE_STREET_FIGURE} question "
        f"the street, or {config.CHAT_NO_VALUE_REPLY}. DEFECT TYPE is one of "
        f"{', '.join(config.DEFECT_STANDARD_DOCUMENTS)} or {config.CHAT_NO_VALUE_REPLY}. ANSWER is your answer from "
        f"the documents, or {config.CHAT_NO_VALUE_REPLY}.",
        "Reply in exactly this layout and nothing else:\n" + "\n".join(f"{label}: ..." for label in LABELS)])
    agent = config.AGENT_NAME_STREET_CARE_ASSISTANT
    fields = parse_labelled_response(call_language_model(prompt, agent)["response_text"], LABELS, agent)
    if any("\n" in fields[label] for label in LABELS[:3]):
        raise ValueError(f"{agent} wrote one of {', '.join(LABELS[:3])} over more than one line")
    question_type, report_value = _value(fields, "QUESTION TYPE").upper(), _value(fields, "REPORT")
    if question_type not in config.CHAT_TYPE_OPTIONS or not report_value:
        raise ValueError(f"{agent} replied with the type {question_type} and the report {report_value!r}")
    if question_type in config.CHAT_ANSWERED_TYPES and re.search(config.REPORT_REFERENCE_IN_TEXT_PATTERN,
                                                                  question_text):
        question_type = config.CHAT_TYPE_REPORT_STATUS
    if question_type in config.CHAT_ANSWERED_TYPES and fields["ANSWER"].lower() == config.CHAT_NO_VALUE_REPLY:
        raise ValueError(f"{agent} gave no answer to a {question_type} question")
    report_reference = street_name = defect_type = None
    if question_type in config.CHAT_REPORT_TYPES and report_value.lower() != config.CHAT_NO_VALUE_REPLY:
        report_reference = convert_report_references(pd.Series([report_value.upper()])).iloc[0]
        report_reference = None if pd.isna(report_reference) else report_reference
        if report_reference and not _named(report_reference, question_text + "\n" + conversation["history_text"]):
            raise ValueError(f"{agent} named {report_reference}, which neither the question nor the conversation "
                             f"names")
    if question_type == config.CHAT_TYPE_STREET_FIGURE:
        street_name = _matching(report_value, STREETS)
    if question_type in (config.CHAT_TYPE_DEFECT_STANDARD, config.CHAT_TYPE_STREET_FIGURE):
        defect_type = _matching(_value(fields, "DEFECT TYPE"), config.DEFECT_STANDARD_DOCUMENTS)
    closest = passages[0]["distance"]
    answered = question_type in config.CHAT_ANSWERED_TYPES and closest <= config.CHAT_MAX_RELEVANCE_DISTANCE
    sources = []
    if answered and question_type == config.CHAT_TYPE_DEFECT_STANDARD:
        sources += ([config.DEFECT_STANDARD_DOCUMENTS[defect_type]] if defect_type
                    else list(config.DEFECT_STANDARD_DOCUMENTS.values()))
    if answered and question_type == config.CHAT_TYPE_STATEMENT_GUIDANCE:
        sources += list(config.SENTENCE_TYPE_GUIDE_DOCUMENTS.values())
    if answered:
        sources += [passage["document_name"] for passage in passages]
    refused = not answered and question_type not in config.CHAT_RECORD_TYPES
    return {"question_type": config.CHAT_TYPE_OUT_OF_SCOPE if refused else question_type,
            "report_reference": report_reference, "street_name": street_name, "defect_type": defect_type,
            "answer": config.CHAT_REFUSAL_MESSAGE if refused else fields["ANSWER"],
            "source_documents": list(dict.fromkeys(sources)), "closest_distance": closest, "was_refused": refused}
