import json
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from backend.services.vector_db_service import VectorDBService
from backend.constants import KnowledgeSource
from config import FAQ_PATH, OPERATOR_KNOWLEDGE_PATH


class KnowledgeManager:
    """
    Manages the lifecycle of knowledge data, including file synchronization,
    vector database updates, and memory caching for API responses.
    """

    def __init__(self):
        self.db = VectorDBService()
        self._faq_cache = {}
        self._operator_cache = []

    @staticmethod
    def _generate_id(text: str) -> str:
        """Creates a stable MD5 hash for a given text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _sync_to_db(self, new_data: Dict[str, Dict[str, Any]], source: KnowledgeSource):
        """
        Compares existing DB IDs with new IDs, removes orphans, and upserts current data.
        """
        new_ids: Set[str] = set(new_data.keys())

        # 1. Get existing IDs from DB for this source
        existing_ids: Set[str] = set(self.db.get_ids_by_metadata({"source": source.value}))

        # 2. Identify and remove IDs no longer present in the source
        ids_to_remove = list(existing_ids - new_ids)
        self.db.delete_by_ids(ids_to_remove)

        # 3. Upsert current items
        self.db.upsert_batch(
            documents=[v["text"] for v in new_data.values()],
            ids=list(new_data.keys()),
            metadatas=[v["metadata"] for v in new_data.values()]
        )

    def load_faq_data(self) -> None:
        """
        Loads data from a file, updates the memory cache, and syncs with Vector DB.
        """
        # Prepare FAQ data
        with open(FAQ_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self._faq_cache = data

            processed_items = {}
            for cat_id, questions in data.get('faq', {}).items():
                for item in questions:
                    text = f"Question: {item['q']} Answer: {item['a']}"
                    item_id = self._generate_id(text)
                    processed_items[item_id] = {
                        "text": text,
                        "metadata": {"source": KnowledgeSource.FAQ.value}
                    }

            self._sync_to_db(processed_items, KnowledgeSource.FAQ)
            print("✅ Knowledge Sync: Vector DB updated from the FAQ source.")

    def load_operator_knowledge(self) -> None:
        """Loads operator knowledge and syncs with Vector DB."""
        if os.path.exists(OPERATOR_KNOWLEDGE_PATH):
            with open(OPERATOR_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                self._operator_cache = json.load(f)

            processed_items = {}
            for item in self._operator_cache:
                text = f"Question: {item['q']} Answer: {item['a']}"
                item_id = self._generate_id(item['q'])  # ID based on question only to allow updates
                processed_items[item_id] = {
                    "text": text,
                    "metadata": {
                        "source": KnowledgeSource.OPERATOR.value,
                        "created_at": item.get('created_at', "N/A")
                    }
                }

            self._sync_to_db(processed_items, KnowledgeSource.OPERATOR)
            print(f"🧠 Operator knowledge synced.")

    def get_categories(self) -> List[Dict[str, Any]]:
        """Returns the cached list of FAQ categories."""
        return self._faq_cache['categories']

    def get_questions_by_category(self, category_id) -> List[Dict[str, Any]]:
        """Returns the cached list of questions for a specific category."""
        return self._faq_cache['faq'][category_id]

    def save_operator_answer(self, question: str, answer: str):
        """Saves a new human answer to a knowledge file."""
        new_entry = {
            "q": question,
            "a": answer,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Load existing, append and save
        if not os.path.exists(OPERATOR_KNOWLEDGE_PATH):
            with open(OPERATOR_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(OPERATOR_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Search if such a question already exists
        existing_idx = next((i for i, item in enumerate(data) if item['q'] == question), None)

        if existing_idx:
            # If found, replace the old answer with a new one
            data[existing_idx] = new_entry
        else:
            # If not found, add it
            data.append(new_entry)

        # Save the updated list back to the file
        with open(OPERATOR_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        doc_id = self._generate_id(question)
        text = f"Question: {question} Answer: {answer}"

        self.db.upsert_batch(
            documents=[text],
            ids=[doc_id],
            metadatas=[{
                "source": KnowledgeSource.OPERATOR.value,
                "created_at": new_entry["created_at"]
            }]
        )

        print(f"🧠 Operator knowledge saved for future use.")

    def get_relevant_context(self, query: str) -> Tuple[Optional[str], float]:
        """Queries the vector database for the most relevant context."""
        return self.db.search(query)
