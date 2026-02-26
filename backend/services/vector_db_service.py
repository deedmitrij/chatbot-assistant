import chromadb
from config import PROJECT_ROOT


class VectorDBService:

    def __init__(self, collection="hotel_knowledge"):
        self.client = chromadb.PersistentClient(path=PROJECT_ROOT / "chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def upsert_batch(self, documents, ids, metadatas):
        if documents:
            self.collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def delete_by_ids(self, ids):
        if ids:
            self.collection.delete(ids=ids)

    def get_ids_by_metadata(self, filter_dict):
        results = self.collection.get(where=filter_dict)
        return results['ids']

    def search(self, query_text, n_results=3, where_filter=None):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results
