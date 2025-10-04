# memory/subsystem.py
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os
from typing import List, Dict, Any


class MemorySubsystem:
    """
    Verwaltet das episodische Langzeitgedächtnis von CAPA.
    Nutzt ein Sentence-Transformer-Modell zur Vektorisierung von Erlebnissen
    und ChromaDB zur persistenten, semantischen Speicherung und Abfrage.
    """
    DB_PATH = "./capa_memory_db"
    COLLECTION_NAME = "capa_memory"
    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self):
        """
        Initialisiert den ChromaDB-Client, die persistente Collection und
        lädt das Sentence-Transformer-Modell.
        """
        print("Initialisiere Gedächtnis-Subsystem...")

        # Dev 3: Lädt das Modell zur semantischen Einbettung.
        self.embedding_model = SentenceTransformer(self.MODEL_NAME)

        # Dev 2: Richtet die persistente Vektordatenbank ein.
        if not os.path.exists(self.DB_PATH):
            os.makedirs(self.DB_PATH)

        self.client = chromadb.PersistentClient(path=self.DB_PATH)
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)

        print(f"Gedächtnis-Subsystem bereit. Datenbank-Einträge: {self.collection.count()}")

    def add_experience(self, text_description: str, metadata: dict):
        """
        Wandelt ein textuelles Erlebnis in einen Vektor um und speichert es
        zusammen mit den Metadaten persistent in der Datenbank.

        Args:
            text_description (str): Die textuelle Beschreibung des Erlebnisses.
            metadata (dict): Ein Dictionary mit kontextuellen Daten (z.B. ASC-Zustand).
        """
        # Dev 3: Wandelt den Text in einen semantischen Vektor um.
        embedding = self.embedding_model.encode(text_description)

        # Dev 2: Generiert eine einzigartige ID und speichert die Daten.
        unique_id = str(uuid.uuid4())

        self.collection.add(
            ids=[unique_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[text_description]
        )
        print(f"Erinnerung hinzugefügt: '{text_description}'")

    def query_relevant_memories(self, query_text: str, n_results: int = 1) -> List[Dict[str, Any]]:
        """
        Sucht nach den n relevantesten Erinnerungen zu einem gegebenen Abfragetext.

        Args:
            query_text (str): Der Text, der die aktuelle Situation beschreibt.
            n_results (int): Die Anzahl der zurückzugebenden, ähnlichsten Erinnerungen.

        Returns:
            A list of dictionaries, where each dictionary contains the 'text'
            and 'metadata' of a relevant memory.
        """
        if self.collection.count() == 0:
            return []

        # Dev 3: Wandelt die Abfrage in einen Vektor um.
        query_embedding = self.embedding_model.encode(query_text)

        # Dev 2: Führt die Ähnlichkeitssuche in der Datenbank durch.
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Bereitet die Ergebnisse in einem sauberen Format auf.
        formatted_results = []
        if results['documents']:
            for text, meta in zip(results['documents'][0], results['metadatas'][0]):
                formatted_results.append({'text': text, 'metadata': meta})

        return formatted_results

    def get_memory_count(self) -> int:
        """Hilfsfunktion zur Überprüfung der Anzahl der Erinnerungen."""
        return self.collection.count()