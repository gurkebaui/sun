import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
import os
from typing import List, Dict, Any


class MemorySubsystem:
    """
    Verwaltet das episodische Langzeitgedächtnis von CAPA.
    FINALE ARCHITEKTUR: Trennt nun explizit zwischen Produktions- und Test-Datenbanken.
    """
    # Der Standard-Pfad für den normalen Betrieb
    DEFAULT_DB_PATH = "./capa_memory_db"

    def __init__(self, db_path: str = None):
        """
        Initialisiert den Client mit einem spezifischen Datenbank-Pfad.
        Wenn kein Pfad angegeben wird, wird der Standard-Produktions-Pfad verwendet.
        """
        self.db_path = db_path if db_path is not None else self.DEFAULT_DB_PATH

        print(f"Initialisiere Gedächtnis-Subsystem am Pfad: {self.db_path}...")

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        # Erlaube das Zurücksetzen nur, wenn wir uns explizit im Test-Modus befinden
        allow_db_reset = "test" in self.db_path

        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=allow_db_reset)
        )
        self.collection = self.client.get_or_create_collection(name="capa_memory")

        print(f"Gedächtnis-Subsystem bereit. Datenbank-Einträge: {self.collection.count()}")

    def reset_database_for_testing(self):
        """
        Eine explizite, gefährliche Funktion, die NUR für Tests verwendet werden darf.
        """
        if "test" not in self.db_path:
            raise PermissionError("Das Zurücksetzen der Datenbank ist nur im Test-Modus erlaubt.")
        print("Setze Test-Datenbank zurück...")
        self.client.reset()

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