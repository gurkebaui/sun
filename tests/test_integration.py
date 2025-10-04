# tests/test_integration.py
import unittest
import torch
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import CAPA_Agent
from pag.model import PAG_Model  # Import bleibt gleich
from asc.core import AffectiveStateCore


# --- Alte Hyperparameter werden entfernt ---

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialisiert den Agenten einmal für alle Tests."""
        # --- KORREKTUR: Instanziiere das PAG_Model auf die neue Art ---
        pag = PAG_Model()  # Kein model_name, verwendet den Default "google/flan-t5-base"
        asc = AffectiveStateCore()
        cls.agent = CAPA_Agent(pag_model=pag, asc=asc)

        # --- KORREKTUR: Input ist jetzt ein String-Prompt ---
        cls.input_prompt = "Erzähle mir von der Schwerkraft."

    def test_c_rest_state(self):
        """Test C: Bestätigt, dass das Modell im Ruhezustand normal arbeitet."""
        print("\n--- Integrationstest C: Ruhezustand ---")
        self.agent.asc.set_state(x=0, y=0)
        output = self.agent.run_inference(self.input_prompt)
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_a_focus_state(self):
        """Test A: Verifiziert, dass bei hohem Arousal (Fokus) anders gearbeitet wird."""
        print("\n--- Integrationstest A: Fokus-Zustand ---")
        self.agent.asc.set_state(x=95, y=0)
        output = self.agent.run_inference(self.input_prompt)
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_b_creativity_state(self):
        """Test B: Weist nach, dass die Ausgabe bei positiver Valenz variabler ist."""
        print("\n--- Integrationstest B: Kreativitäts-Zustand ---")
        self.agent.asc.set_state(x=0, y=100)  # Hohe positive Valenz -> hohe Temperatur

        outputs = {self.agent.run_inference(self.input_prompt) for _ in range(3)}

        print(f"Anzahl einzigartiger Ausgaben bei 3 Durchläufen: {len(outputs)}")
        self.assertTrue(len(outputs) > 1, "Die Ausgaben sollten bei hoher Temperatur variieren.")


if __name__ == '__main__':
    unittest.main()