import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import config


def _embedding():
    return SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)


def _has_store_file(store_path):
    return os.path.isfile(os.path.join(store_path, config.VECTOR_STORE_FILE_NAME))


def build_vector_index(chunk_records, store_path):
    if os.path.isfile(store_path) or (os.path.isdir(store_path) and os.listdir(store_path)
                                      and not _has_store_file(store_path)):
        raise ValueError(f"{store_path} is not a folder holding a vector store")
    embedding = _embedding()
    client = chromadb.PersistentClient(path=store_path)
    if config.VECTOR_COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(config.VECTOR_COLLECTION_NAME)
    search_breadth = max(len(chunk_records), 1)
    collection = client.create_collection(
        config.VECTOR_COLLECTION_NAME, embedding_function=embedding,
        configuration={"hnsw": {"space": config.VECTOR_DISTANCE_METRIC, "ef_construction": search_breadth,
                                "ef_search": search_breadth}})
    collection.add(ids=[record["chunk_id"] for record in chunk_records],
                   documents=[record["chunk_text"] for record in chunk_records],
                   metadatas=[{key: record[key] for key in ("document_name", "section_number", "passage_number",
                                                            "carried_characters")} for record in chunk_records])
    return collection.count()


def open_vector_index(store_path):
    if not _has_store_file(store_path):
        raise ValueError(f"{store_path} holds no vector store")
    client = chromadb.PersistentClient(path=store_path)
    if config.VECTOR_COLLECTION_NAME not in [collection.name for collection in client.list_collections()]:
        raise ValueError(f"{store_path} holds no collection {config.VECTOR_COLLECTION_NAME}")
    return client.get_collection(config.VECTOR_COLLECTION_NAME, embedding_function=_embedding())


def read_document_section(collection, document_name, section_number):
    found = collection.get(where={"$and": [{"document_name": document_name}, {"section_number": section_number}]},
                           include=["documents", "metadatas"])
    if not found["ids"]:
        raise ValueError(config.NO_SECTION_PASSAGES_MESSAGE.format(section_number=section_number,
                                                                   document_name=document_name))
    passages = sorted(zip(found["metadatas"], found["documents"]), key=lambda pair: pair[0]["passage_number"])
    text = passages[0][1]
    for metadata, passage in passages[1:]:
        text += " " + passage[metadata["carried_characters"]:]
    return text


def search_documents(collection, query_texts):
    found = collection.query(query_texts=query_texts, n_results=config.CHAT_RETRIEVAL_RESULT_COUNT,
                             where={"document_name": {"$nin": config.CHAT_EXCLUDED_DOCUMENTS}},
                             include=["documents", "metadatas", "distances"])
    return [[{"chunk_id": chunk_id, "chunk_text": text, "document_name": metadata["document_name"],
              "distance": float(distance)}
             for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances)]
            for ids, texts, metadatas, distances in zip(found["ids"], found["documents"], found["metadatas"],
                                                        found["distances"])]
