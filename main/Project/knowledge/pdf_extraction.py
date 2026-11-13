import os

from pypdf import PdfReader

from config import config


def extract_text_from_directory(docs_directory):
    names = sorted(name for name in os.listdir(docs_directory) if name.lower().endswith(config.PDF_FILE_SUFFIX))
    if not names:
        raise ValueError(f"{docs_directory} holds no PDF document")
    entries = []
    for name in names:
        pages = [page.extract_text() or "" for page in PdfReader(os.path.join(docs_directory, name)).pages]
        if not pages or any(not page.strip() for page in pages):
            raise ValueError(f"{name} has no pages or a page that gives up no text")
        entries.append({"document_name": name, "document_text": "\n".join(pages)})
    return entries
