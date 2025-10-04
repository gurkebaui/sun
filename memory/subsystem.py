# memory/subsystem.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
import os
from typing import List, Dict, Any


class MemorySubsystem:
    """
    Verwaltet das episodische Langzeitgedächtnis von CAPA.
    Dies ist die finale, vollständige und korrigierte Version.
    """
    DB_PATH = "./capa_memory_db"
    COLLECTION_NAME = "capa_memory"
    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self):
        print("Initialisiere Gedächtnis-Subsystem...")

        self.embedding_model = SentenceTransformer(self.MODEL_NAME)

        if not os.path.exists(self.DB_PATH):
            os.makedirs(self.DB_PATH)

        self.client = chromadb.PersistentClient(
            path=self.DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)

        print(f"Gedächtnis-Subsystem bereit. Datenbank-Einträge: {self.collection.count()}")

    def add_experience(self, text_description: str, metadata: dict):
        embedding = self.embedding_model.encode(text_description).tolist()
        unique_id = str(uuid.uuid4())

        self.collection.add(
            ids=[unique_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text_description]
        )
        print(f"Erinnerung hinzugefügt: '{text_description}'")

    def query_relevant_memories(self, query_text: str, n_results: int = 1) -> List[Dict[str, Any]]:
        """
        Sucht nach den n relevantesten Erinnerungen zu einem gegebenen Abfragetext.
        DIESE METHODE WURDE WIEDERHERGESTELLT.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        formatted_results = []
        if results['documents']:
            for text, meta in zip(results['documents'][0], results['metadatas'][0]):
                formatted_results.append({'text': text, 'metadata': meta})

        return formatted_results

    def get_latest_memories(self, n_results: int) -> List[Dict[str, Any]]:
        """
        Ruft die n zuletzt hinzugefügten Erinnerungen aus der Datenbank ab.
        """
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)

        results = self.collection.get(
            limit=count,
            include=["metadatas", "documents"]
        )

        latest_docs = results['documents'][-n_results:]
        latest_metas = results['metadatas'][-n_results:]

        formatted_results = []
        for text, meta in zip(latest_docs, latest_metas):
            formatted_results.append({'text': text, 'metadata': meta})

        return formatted_results

    def get_memory_count(self) -> int:
        return self.collection.count()

    def shutdown(self):
        """
        Gibt alle vom ChromaDB-Client gehaltenen Ressourcen explizit frei.
        """
        print("Fahre Gedächtnis-Subsystem herunter und gebe Ressourcen frei...")
        self.client.reset()