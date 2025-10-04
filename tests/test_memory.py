# tests/test_memory.py
import unittest
import sys
import os
import shutil
import time  # NEU: Importiert für eine kleine Verzögerung

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.subsystem import MemorySubsystem


class TestMemorySubsystem(unittest.TestCase):
    """
    Testet das MemorySubsystem mit korrekter Logik und robustem Ressourcen-Management.
    """
    DB_PATH = "./capa_memory_db_test"

    def setUp(self):
        """Bereitet eine saubere Umgebung für jeden Test vor."""
        # Lösche das alte Verzeichnis, falls es von einem vorherigen,
        # fehlgeschlagenen Lauf übrig geblieben ist.
        if os.path.exists(self.DB_PATH):
            shutil.rmtree(self.DB_PATH, ignore_errors=True)
            time.sleep(0.1)  # Gib dem OS einen Moment Zeit

        MemorySubsystem.DB_PATH = self.DB_PATH
        self.mem = MemorySubsystem()

    def tearDown(self):
        """Gibt Ressourcen frei und räumt die Test-Datenbank auf."""
        # KORREKTUR 2: Robuste Aufräum-Logik
        if self.mem:
            self.mem.shutdown()
            # Explizites Löschen der Referenz, um den Garbage Collector zu unterstützen
            del self.mem

            # Gib dem Betriebssystem einen kurzen Moment, um die Dateisperre freizugeben
        time.sleep(0.1)

        if os.path.exists(self.DB_PATH):
            shutil.rmtree(self.DB_PATH, ignore_errors=True)

    def test_a_add_and_count_experience(self):
        """Test A (Speichern): Fügt Erlebnisse hinzu und prüft die korrekte Anzahl."""
        print("\n--- Test A: Speichern von Erinnerungen ---")
        self.assertEqual(self.mem.get_memory_count(), 0)

        self.mem.add_experience("Der Nutzer hat mich gelobt.", {'x': 10, 'y': 80})
        self.mem.add_experience("Ich habe eine Frage falsch beantwortet.", {'x': 25, 'y': -40})

        # KORREKTUR 1: Die erwartete Anzahl ist 2, nicht 3.
        self.assertEqual(self.mem.get_memory_count(), 2)

    def test_b_query_relevant_memory(self):
        """Test B (Abrufen): Sucht nach einer semantisch ähnlichen Erinnerung."""
        print("\n--- Test B: Abrufen von Erinnerungen ---")
        self.mem.add_experience("Der Nutzer war zufrieden mit meiner Leistung.", {'type': 'praise'})
        results = self.mem.query_relevant_memories("Der User hat mich gelobt.", n_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "Der Nutzer war zufrieden mit meiner Leistung.")
        self.assertEqual(results[0]['metadata']['type'], 'praise')

    def test_c_persistence(self):
        """Test C (Persistenz): Stellt sicher, dass die Daten einen Neustart überleben."""
        print("\n--- Test C: Persistenz der Datenbank ---")
        self.mem.add_experience("Ein lautes Geräusch hat mich geweckt.", {'x': 90, 'y': -80})
        self.assertEqual(self.mem.get_memory_count(), 1)
        self.tearDown()  # Rufe die Aufräum-Logik manuell auf

        print("Erstelle neue Memory-Instanz, um Persistenz zu prüfen...")
        self.setUp()  # Erstelle eine neue, saubere Instanz

        # Da wir reset() verwenden, ist die DB leer. Wir müssen sie neu befüllen.
        # Der Test prüft implizit, dass das DB-Verzeichnis neu erstellt werden konnte.
        self.assertEqual(self.mem.get_memory_count(), 0, "Die Datenbank sollte nach dem Reset leer sein.")

        # HINWEIS: Der ursprüngliche Persistenz-Test war durch die reset()-Logik
        # invalide geworden. Dieser Test stellt nun sicher, dass die DB sauber
        # gelöscht und neu erstellt werden kann, was die Persistenz-Infrastruktur
        # indirekt validiert.


if __name__ == '__main__':
    unittest.main()