# tests/test_memory.py
import unittest
import sys
import os
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.subsystem import MemorySubsystem


class TestMemorySubsystem(unittest.TestCase):
    """
    Testet die Kernfunktionalität des MemorySubsystems, einschließlich
    Speichern, Abrufen und Datenpersistenz.
    """
    DB_PATH = "./capa_memory_db_test"  # Verwende eine separate Test-Datenbank

    def setUp(self):
        """Stellt sicher, dass vor jedem Test eine saubere Datenbank existiert."""
        if os.path.exists(self.DB_PATH):
            shutil.rmtree(self.DB_PATH)

        # Überschreibe den Klassen-Pfad für die Test-Instanz
        MemorySubsystem.DB_PATH = self.DB_PATH

    def tearDown(self):
        """Räumt die Test-Datenbank nach den Tests auf."""
        if os.path.exists(self.DB_PATH):
            shutil.rmtree(self.DB_PATH)

    def test_a_add_and_count_experience(self):
        """Test A (Speichern): Fügt Erlebnisse hinzu und prüft die Anzahl."""
        print("\n--- Test A: Speichern von Erinnerungen ---")
        mem = MemorySubsystem()
        self.assertEqual(mem.get_memory_count(), 0)

        mem.add_experience("Der Nutzer hat mich für eine gute Zusammenfassung gelobt.", {'x': 10, 'y': 80})
        mem.add_experience("Ich habe eine Frage über den Mond falsch beantwortet.", {'x': 25, 'y': -40})
        mem.add_experience("Ein lautes Geräusch hat mich aus dem Schlaf gerissen.", {'x': 90, 'y': -80})

        self.assertEqual(mem.get_memory_count(), 3)

    def test_b_query_relevant_memory(self):
        """Test B (Abrufen): Sucht nach einer semantisch ähnlichen Erinnerung."""
        print("\n--- Test B: Abrufen von Erinnerungen ---")
        mem = MemorySubsystem()
        mem.add_experience("Der Nutzer hat mich für eine gute Zusammenfassung gelobt.",
                           {'x': 10, 'y': 80, 'type': 'praise'})
        mem.add_experience("Ich habe eine Frage über den Mond falsch beantwortet.",
                           {'x': 25, 'y': -40, 'type': 'failure'})

        # Die Abfrage ist semantisch ähnlich, nicht identisch.
        query = "Der User war sehr zufrieden mit meiner Leistung."
        results = mem.query_relevant_memories(query, n_results=1)

        self.assertEqual(len(results), 1)
        retrieved_memory = results[0]

        # Verifiziere den Inhalt und die Metadaten
        self.assertEqual(retrieved_memory['text'], "Der Nutzer hat mich für eine gute Zusammenfassung gelobt.")
        self.assertEqual(retrieved_memory['metadata']['y'], 80)
        self.assertEqual(retrieved_memory['metadata']['type'], 'praise')

    def test_c_persistence(self):
        """Test C (Persistenz): Stellt sicher, dass die Daten einen Neustart überleben."""
        print("\n--- Test C: Persistenz der Datenbank ---")
        # Phase 1: Daten hinzufügen und Objekt zerstören
        mem1 = MemorySubsystem()
        mem1.add_experience("Ein lautes Geräusch hat mich aus dem Schlaf gerissen.", {'x': 90, 'y': -80})
        self.assertEqual(mem1.get_memory_count(), 1)
        del mem1

        # Phase 2: Neues Objekt erstellen und prüfen, ob die Daten noch da sind
        print("Erstelle neue Memory-Instanz, um Persistenz zu prüfen...")
        mem2 = MemorySubsystem()
        self.assertEqual(mem2.get_memory_count(), 1,
                         "Die Daten sollten nach der Neu-Initialisierung noch vorhanden sein.")

        # Prüfe, ob die alten Daten abgerufen werden können
        results = mem2.query_relevant_memories("plötzlicher Lärm", n_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "Ein lautes Geräusch hat mich aus dem Schlaf gerissen.")
        self.assertEqual(results[0]['metadata']['y'], -80)


if __name__ == '__main__':
    unittest.main()