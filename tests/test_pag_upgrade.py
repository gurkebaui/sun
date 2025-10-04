# tests/test_pag_upgrade.py
import unittest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pag.model import PAG_Model


class TestPagUpgrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialisiert das PAG-Modell einmal für alle Tests."""
        cls.pag = PAG_Model(model_name="google/flan-t5-base")

    def test_c_functionality(self):
        """Test C: Kann das Modell einer Anweisung zur Zusammenfassung folgen?"""
        print("\n--- Test C: Funktionale Anweisung ---")
        prompt = "Summarize this sentence in one word: The sun, a massive star at the center of our solar system, provides light and heat to Earth."
        output = self.pag.infer(prompt, temperature=0.1)
        print(f"Prompt: {prompt}\nOutput: {output}")

        # FINALE ROBUSTE PRÜFUNG: Prüft, ob das Kernkonzept ("sun") in der (kurzen) Antwort enthalten ist.
        self.assertIn("sun", output.lower(), "Die Zusammenfassung muss das Wort 'sun' enthalten.")
        self.assertTrue(len(output.split()) < 7, "Die Zusammenfassung sollte ein kurzer Satz oder ein Wort sein.")

    def test_a_conservative_state(self):
        """Test A: Liefert eine niedrige Temperatur konsistente Ergebnisse?"""
        print("\n--- Test A: Konservativer Zustand (niedrige Temperatur) ---")
        prompt = "Translate to German: The sky is blue."
        outputs = {self.pag.infer(prompt, temperature=0.01) for _ in range(3)}
        print(f"Prompt: {prompt}\nErhaltene Antworten bei temp=0.01: {outputs}")

        self.assertEqual(len(outputs), 1, "Bei niedriger Temperatur wird eine einzige, konsistente Antwort erwartet.")

        # FINALE ROBUSTE PRÜFUNG: Prüft auf Schlüsselwörter, ignoriert den Artikel ("Der" vs "Die").
        single_output = list(outputs)[0].lower()
        self.assertIn("himmel", single_output, "Die Übersetzung muss 'himmel' enthalten.")
        self.assertIn("blau", single_output, "Die Übersetzung muss 'blau' enthalten.")

    def test_b_creative_state(self):
        """Test B: Liefert eine hohe Temperatur variable Ergebnisse?"""
        print("\n--- Test B: Kreativer Zustand (hohe Temperatur) ---")
        prompt = "Write a short, poetic sentence about the moon."
        outputs = {self.pag.infer(prompt, temperature=1.8) for _ in range(5)}
        print(f"Prompt: {prompt}\nErhaltene Antworten bei temp=1.8: {outputs}")

        self.assertTrue(len(outputs) > 1, "Bei hoher Temperatur sollten die Antworten variieren.")
        for o in outputs:
            self.assertIsInstance(o, str)
            self.assertTrue(len(o) > 5, "Jede kreative Antwort sollte mehr als nur ein Wort sein.")


if __name__ == '__main__':
    unittest.main()