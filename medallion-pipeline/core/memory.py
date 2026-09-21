import chromadb

from core.config import CHROMA_DIR


def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name="idamp_memory"):
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=None,
    )


def store_document(doc_id, text, metadata=None):
    try:
        collection = get_collection()
        values = {"ids": [doc_id], "documents": [text]}
        if metadata is not None:
            values["metadatas"] = [metadata]
        collection.upsert(**values)
    except Exception:
        pass


def query_memory(query_text, n_results=5):
    try:
        collection = get_collection()
        results = collection.get(limit=n_results)
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        return [
            {
                "id": doc_id,
                "document": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else None,
            }
            for index, doc_id in enumerate(ids)
        ]
    except Exception:
        return []