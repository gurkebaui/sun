# tests/test_modulation.py
import unittest
import sys
import os
import random

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modulation.functions import (
    modulate_temperature,
    modulate_attention_scores,
    calculate_layer_drop_rate
)


class TestModulationFunctions(unittest.TestCase):
    """
    Test-Suite für die Modulationsfunktionen, die ASC und PAG verbinden.
    """

    def test_modulate_temperature(self):
        """Testet die Temperatur-Modulation basierend auf y- und x-Werten."""
        # --- Tests mit neutralem x-Wert (x=0), LOGIK INVERTIERT ---
        # y > 0 (Belohnung) -> höhere Temperatur
        self.assertAlmostEqual(modulate_temperature(0.0, 100.0), 2.0)
        self.assertAlmostEqual(modulate_temperature(0.0, 50.0), 1.5)

        # y = 0 (neutral) -> neutrale Temperatur
        self.assertAlmostEqual(modulate_temperature(0.0, 0.0), 1.0)

        # y < 0 (Bestrafung) -> niedrigere Temperatur
        self.assertAlmostEqual(modulate_temperature(0.0, -50.0), 0.5)
        # Formel ergibt 0.0, Clamp greift -> 0.01
        self.assertAlmostEqual(modulate_temperature(0.0, -100.0), 0.01)

        # --- Tests für die Arousal-basierte Logik mit neuer Basis-Temperatur ---
        # Fall 1: Hohe Belohnung (base_temp=2.0), Fokus x=90
        # Erwartet: temp zwischen 2.0 und 0.3
        self.assertAlmostEqual(modulate_temperature(90.0, 100.0), 0.64)  # 2.0 - (2.0-0.3)*0.8

        # Fall 2: Maximales Arousal (x=100) erzwingt temp=0.3, unabhängig von y
        self.assertAlmostEqual(modulate_temperature(100.0, 100.0), 0.3)
        self.assertAlmostEqual(modulate_temperature(100.0, -100.0), 0.3)

    def test_modulate_attention_scores(self):
        """Testet die Attention-Score-Modulation basierend auf dem x-Wert."""
        base_scores = [10.0, 20.0, 5.0]

        # x im normalen Bereich [0, 50] -> keine Änderung
        self.assertListEqual(modulate_attention_scores(25, base_scores), base_scores)
        self.assertListEqual(modulate_attention_scores(0, base_scores), base_scores)

        # x > 50 (hoher Fokus) -> Schärfung
        sharpened_scores = modulate_attention_scores(75, base_scores)  # Faktor = 1.5
        self.assertListEqual(sharpened_scores, [15.0, 30.0, 7.5])

        max_sharpened_scores = modulate_attention_scores(100, base_scores)  # Faktor = 2.0
        self.assertListEqual(max_sharpened_scores, [20.0, 40.0, 10.0])

        # x < 0 (Verwirrung) -> Rauschen
        random.seed(42)  # Für reproduzierbare Testergebnisse
        noisy_scores = modulate_attention_scores(-50, base_scores)
        self.assertNotEqual(noisy_scores, base_scores)
        self.assertEqual(len(noisy_scores), len(base_scores))
        # Prüfen, ob die Werte sich im erwarteten Rahmen geändert haben
        # max_noise_amplitude = (20-5)*0.1 = 1.5; noise_strength = 0.5 * 1.5 = 0.75
        # Erwarteter Bereich für den ersten Score: 10 +/- 0.75
        self.assertTrue(9.25 <= noisy_scores[0] <= 10.75)

    def test_calculate_layer_drop_rate(self):
        """Testet die Layer-Drop-Rate basierend auf dem x-Wert."""
        # x < 90 -> Rate ist 0.0
        self.assertEqual(calculate_layer_drop_rate(-100), 0.0)
        self.assertEqual(calculate_layer_drop_rate(0), 0.0)
        self.assertEqual(calculate_layer_drop_rate(89.9), 0.0)

        # x >= 90 -> Rate steigt linear
        self.assertEqual(calculate_layer_drop_rate(90), 0.0)
        self.assertAlmostEqual(calculate_layer_drop_rate(95), 0.25)
        self.assertAlmostEqual(calculate_layer_drop_rate(100), 0.5)

        # Testet den Clamp bei Werten > 100
        self.assertAlmostEqual(calculate_layer_drop_rate(110), 0.5)


if __name__ == '__main__':
    unittest.main()