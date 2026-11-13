import sys

from config import config
from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.vector_index import build_vector_index, open_vector_index
from tools.retrieval_tools import read_inspection_records


def main():
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    sections = split_documents_into_sections(documents)
    stored = build_vector_index(build_chunk_records(sections), config.VECTOR_STORE_DIRECTORY)
    print(f"{len(documents)} documents, {len(sections)} sections, {stored} passages indexed in "
          f"{config.VECTOR_STORE_DIRECTORY}")
    collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
    print("what an inspection records under each defect standard, read back from the index")
    for defect_type, document_name in config.DEFECT_STANDARD_DOCUMENTS.items():
        print(f"  {defect_type}  {document_name}")
        print(f"      {read_inspection_records(collection, defect_type)}")


if __name__ == "__main__":
    sys.exit(main())
