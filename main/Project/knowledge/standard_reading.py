import re
from decimal import Decimal

from config import config
from knowledge.chunk_splitting import split_documents_into_sections

TEXT_PARTS = (config.REQUIREMENT_NAME_PART, config.REQUIREMENT_SENTENCE_PART)


def _read_requirement(document_name, text):
    fits = [(kind, re.fullmatch(pattern, text)) for kind, pattern in config.REQUIREMENT_SENTENCE_PATTERNS.items()]
    fits = [(kind, match) for kind, match in fits if match]
    if len(fits) != 1:
        raise ValueError(f"{document_name} states a requirement no form reads: {text}")
    kind, match = fits[0]
    rules = {part: Decimal(value) for part, value in match.groupdict().items()
             if part not in TEXT_PARTS and value is not None}
    return {"requirement_kind": kind, "requirement_name": match.group(config.REQUIREMENT_NAME_PART),
            "requirement_sentence": match.group(config.REQUIREMENT_SENTENCE_PART), "rules": rules}


def read_defect_standard(defect_type, document_entries):
    if defect_type not in config.DEFECT_STANDARD_DOCUMENTS:
        raise ValueError(config.NO_STANDARD_CONFIGURED_MESSAGE.format(defect_type=defect_type))
    document_name = config.DEFECT_STANDARD_DOCUMENTS[defect_type]
    entries = [entry for entry in document_entries if entry["document_name"] == document_name]
    if len(entries) != 1:
        raise ValueError(f"{document_name} is not among the documents exactly once")
    sections = {section["section_number"]: section["section_text"]
                for section in split_documents_into_sections(entries)}
    stated = re.findall(config.DEFECT_TYPE_PATTERN, sections.get(config.DEFECT_TYPE_SECTION_NUMBER, ""))
    if stated != [defect_type]:
        raise ValueError(f"{document_name} does not state the defect type {defect_type} alone")
    body = re.fullmatch(config.REQUIREMENTS_SECTION_PATTERN, sections.get(config.REQUIREMENTS_SECTION_NUMBER, ""))
    if body is None:
        raise ValueError(f"{document_name} has no requirements section laid out as the rules read it")
    pieces = re.split(config.REQUIREMENT_NUMBER_PATTERN, body.group(1))
    numbers, texts = pieces[1::2], pieces[2::2]
    if pieces[0].strip() or [int(number) for number in numbers] != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{document_name} does not number its requirements from 1 in order")
    requirements = [_read_requirement(document_name, text.strip()) for text in texts]
    kinds = [requirement["requirement_kind"] for requirement in requirements]
    if len(kinds) != len(set(kinds)):
        raise ValueError(f"{document_name} states a requirement kind twice")
    return {"defect_type": defect_type, "document_name": document_name, "requirements": requirements}
