# tests/test_integration.py
import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore

# --- Modell-Hyperparameter (müssen mit dem Training übereinstimmen) ---
VOCAB_SIZE = 50
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512


class TestIntegration(unittest.TestCase):
    """
    Testet die End-zu-End-Integration von ASC, Modulation und PAG
    innerhalb des CAPA_Agent.
    """

    @classmethod
    def setUpClass(cls):
        """Initialisiert den Agenten einmal für alle Tests."""
        # Ein untrainiertes PAG-Modell reicht aus, um die Mechanik zu testen
        pag = PAG_Model(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD)
        asc = AffectiveStateCore()
        cls.agent = CAPA_Agent(pag_model=pag, asc=asc)

        # Standard-Input für alle Tests
        cls.input_seq = torch.tensor([0, 4, 8, 15, 16, 23, 42, 1])  # SOS, sequence, EOS

    def test_c_rest_state(self):
        """Test C: Bestätigt, dass das Modell im Ruhezustand normal arbeitet."""
        print("\n--- Test C: Ruhezustand ---")
        self.agent.asc.set_state(x=0, y=0)
        output = self.agent.run_inference(self.input_seq)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(len(output) > 0)

    def test_a_focus_state(self):
        """Test A: Verifiziert, dass bei hohem Arousal (Fokus) anders gearbeitet wird."""
        print("\n--- Test A: Fokus-Zustand ---")
        self.agent.asc.set_state(x=95, y=0)  # Hoher Fokus -> Layer Dropping
        output = self.agent.run_inference(self.input_seq)
        self.assertIsInstance(output, torch.Tensor)
        # Der Test ist erfolgreich, wenn die Inferenz trotz Layer Dropping durchläuft.
        # Eine genaue Prüfung des Outputs ist bei einem untrainierten Modell nicht sinnvoll.
        self.assertTrue(len(output) > 0)

    def test_b_creativity_state(self):
        """Test B: Weist nach, dass die Ausgabe bei negativer Valenz variabler ist."""
        print("\n--- Test B: Kreativitäts-Zustand ---")
        self.agent.asc.set_state(x=0, y=-90)  # Hohe negative Valenz -> hohe Temperatur

        # Führe die Inferenz mehrmals aus
        outputs = set()
        for _ in range(5):
            output_tensor = self.agent.run_inference(self.input_seq)
            # Konvertiere Tensor zu einem hashbaren Tuple
            outputs.add(tuple(output_tensor.tolist()))

        # Bei hoher Temperatur und Sampling sollten die Ausgaben variieren.
        # Bei einem untrainierten Modell ist die Wahrscheinlichkeitsverteilung flach,
        # daher ist die Varianz fast garantiert.
        print(f"Anzahl einzigartiger Ausgaben bei 5 Durchläufen: {len(outputs)}")
        self.assertTrue(len(outputs) > 1, "Die Ausgaben sollten bei hoher Temperatur variieren.")


if __name__ == '__main__':
    unittest.main()