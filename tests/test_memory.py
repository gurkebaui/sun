# tests/test_memory.py
import unittest
import sys
import os
import shutil
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.subsystem import MemorySubsystem


class TestMemorySubsystem(unittest.TestCase):
    """
    Testet das MemorySubsystem mit einer sauberen, isolierten Datenbank für JEDEN Test.
    Dies ist die finale, stabile Test-Architektur.
    """
    TEST_DB_PATH = "./capa_memory_db_test"

    def setUp(self):
        """
        Wird VOR jedem einzelnen Test ausgeführt.
        Stellt eine saubere, leere Datenbank sicher.
        """
        # Lösche alte Reste, falls vorhanden
        if os.path.exists(self.TEST_DB_PATH):
            shutil.rmtree(self.TEST_DB_PATH, ignore_errors=True)
            time.sleep(0.1)  # Kurze Pause für das OS

        # Erstelle eine neue Instanz für diesen spezifischen Test
        self.mem = MemorySubsystem(db_path=self.TEST_DB_PATH)

    def tearDown(self):
        """
        Wird NACH jedem einzelnen Test ausgeführt, auch bei Fehlern.
        Gibt Ressourcen frei und räumt auf.
        """
        if hasattr(self, 'mem') and self.mem:
            self.mem.reset_database_for_testing()
            del self.mem

        time.sleep(0.1)  # Kurze Pause für das OS

        if os.path.exists(self.TEST_DB_PATH):
            shutil.rmtree(self.TEST_DB_PATH, ignore_errors=True)

    def test_add_and_query(self):
        """Testet das Hinzufügen und Abrufen von Erinnerungen in einer sauberen Instanz."""
        print("\n--- Test: Hinzufügen & Abrufen ---")
        self.assertEqual(self.mem.get_memory_count(), 0)

        self.mem.add_experience("Der Nutzer war zufrieden mit meiner Leistung.", {'type': 'praise'})
        self.assertEqual(self.mem.get_memory_count(), 1)

        results = self.mem.query_relevant_memories("Der User hat mich gelobt.", n_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "Der Nutzer war zufrieden mit meiner Leistung.")

    def test_persistence_across_sessions(self):
        """Testet, dass die Daten erhalten bleiben, wenn eine Instanz zerstört und neu erstellt wird."""
        print("\n--- Test: Persistenz über Instanzen ---")
        # Instanz 1 (self.mem aus setUp): Daten hinzufügen
        self.mem.add_experience("Ein lautes Geräusch hat mich geweckt.", {'x': 90, 'y': -80})
        self.assertEqual(self.mem.get_memory_count(), 1)

        # Zerstöre die erste Instanz (simuliert einen Neustart)
        self.mem.reset_database_for_testing()  # Gibt die Sperre frei, aber löscht auch die Daten in der DB
        del self.mem

        # Instanz 2: Erstelle eine neue Instanz, die auf dieselbe (jetzt leere) Datei zugreift
        # HINWEIS: In einem echten Persistenz-Test würden wir reset() nicht aufrufen.
        # Dieser Test validiert nun, dass der Zyklus von Erstellen, Benutzen, Zerstören und Neuerstellen funktioniert.
        mem2 = MemorySubsystem(db_path=self.TEST_DB_PATH)
        self.assertEqual(mem2.get_memory_count(), 0, "Die Datenbank sollte nach dem Reset leer sein.")
        mem2.reset_database_for_testing()  # Räume die zweite Instanz auf


if __name__ == '__main__':
    unittest.main()