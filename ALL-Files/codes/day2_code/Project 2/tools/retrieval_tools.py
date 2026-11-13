from config import config
from knowledge.vector_index import read_document_section


def _read_section(collection, document_name, section_number, section_title):
    text = read_document_section(collection, document_name, section_number)
    heading = f"{section_number}. {section_title} "
    if not text.startswith(heading):
        raise ValueError(config.SECTION_TITLE_MISMATCH_MESSAGE.format(
            section_number=section_number, document_name=document_name, section_title=section_title))
    return text[len(heading):]


def _standard_document(defect_type):
    if defect_type not in config.DEFECT_STANDARD_DOCUMENTS:
        raise ValueError(config.NO_STANDARD_CONFIGURED_MESSAGE.format(defect_type=defect_type))
    return config.DEFECT_STANDARD_DOCUMENTS[defect_type]


def read_reviewer_checks(collection, sentence_type):
    if sentence_type not in config.SENTENCE_TYPE_GUIDE_DOCUMENTS:
        raise ValueError(config.NO_GUIDE_CONFIGURED_MESSAGE.format(sentence_type=sentence_type))
    return _read_section(collection, config.SENTENCE_TYPE_GUIDE_DOCUMENTS[sentence_type],
                         config.REVIEWER_CHECKS_SECTION_NUMBER, config.REVIEWER_CHECKS_SECTION_TITLE)


def read_inspection_records(collection, defect_type):
    return _read_section(collection, _standard_document(defect_type), config.INSPECTION_RECORDS_SECTION_NUMBER,
                         config.INSPECTION_RECORDS_SECTION_TITLE)


def read_works_order_guidance(collection, defect_type):
    return _read_section(collection, _standard_document(defect_type), config.WORKS_ORDER_SECTION_NUMBER,
                         config.WORKS_ORDER_SECTION_TITLE)


def read_resident_update_guidance(collection, defect_type):
    return _read_section(collection, _standard_document(defect_type), config.RESIDENT_UPDATE_SECTION_NUMBER,
                         config.RESIDENT_UPDATE_SECTION_TITLE)


def read_policy_section(collection, section_number, section_title):
    return _read_section(collection, config.HIGHWAYS_POLICY_DOCUMENT, section_number, section_title)
