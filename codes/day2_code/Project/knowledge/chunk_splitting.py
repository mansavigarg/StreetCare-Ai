import re

from config import config


def _single_spaced(text):
    return re.sub(config.WHITESPACE_RUN_PATTERN, " ", text).strip()


def split_documents_into_sections(document_entries):
    sections = []
    for entry in document_entries:
        name, text = entry["document_name"], entry["document_text"]
        if len(re.findall(config.DOCUMENT_HEADER_PATTERN, text)) != 1:
            raise ValueError(f"{name} does not carry exactly one header line")
        headings = list(re.finditer(config.SECTION_HEADING_PATTERN, text))
        numbers = [int(heading.group(1)) for heading in headings]
        if not numbers or numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{name} is not divided into sections numbered from 1, in order")
        ends = [heading.start() for heading in headings[1:]] + [len(text)]
        for heading, number, end in zip(headings, numbers, ends):
            sections.append({"document_name": name, "section_number": number,
                             "section_text": _single_spaced(text[heading.start():end])})
    return sections


def build_chunk_records(section_entries):
    records, passage_numbers = [], {}
    for section in section_entries:
        name = section["document_name"]
        sentences = re.split(config.SENTENCE_BOUNDARY_PATTERN, section["section_text"])
        previous, position = [], 0
        while position < len(sentences):
            carried = []
            for sentence in reversed(previous):
                if len(" ".join([sentence] + carried)) > config.CHUNK_OVERLAP_CHARACTERS:
                    break
                carried.insert(0, sentence)
            while carried and len(" ".join(carried + [sentences[position]])) > config.CHUNK_SIZE_CHARACTERS:
                carried.pop(0)
            parts = carried + [sentences[position]]
            position += 1
            while position < len(sentences) and len(" ".join(parts + [sentences[position]])) <= config.CHUNK_SIZE_CHARACTERS:
                parts.append(sentences[position])
                position += 1
            number = passage_numbers.get(name, 0)
            passage_numbers[name] = number + 1
            records.append({"chunk_id": f"{name}#{number}", "chunk_text": " ".join(parts), "document_name": name,
                            "section_number": section["section_number"], "passage_number": number,
                            "carried_characters": len(" ".join(carried)) + 1 if carried else 0})
            previous = parts
    return records
