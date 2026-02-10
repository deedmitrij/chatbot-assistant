import chromadb


class VectorDBService:

    def __init__(self, storage_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = self.client.get_or_create_collection(name="hotel_knowledge")

    def upsert_batch(self, documents, ids, metadatas):
        if documents:
            self.collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def delete_by_ids(self, ids):
        if ids:
            self.collection.delete(ids=ids)

    def get_ids_by_metadata(self, filter_dict):
        results = self.collection.get(where=filter_dict)
        return results['ids']

    def search(self, query_text, n_results=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        context = "\n---\n".join(results['documents'][0])
        nearest_distance = results['distances'][0][0]

        return context, nearest_distance
