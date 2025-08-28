# tests/test_asc.py
import unittest
import sys
import os

# Füge das Hauptverzeichnis zum Python-Pfad hinzu, damit 'asc' importiert werden kann
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asc.core import AffectiveStateCore

class TestAffectiveStateCore(unittest.TestCase):
    """
    Test-Suite für die AffectiveStateCore Klasse.
    Stellt sicher, dass die Zustandsverwaltung und die Clamping-Logik
    korrekt funktionieren.
    """

    def test_initialization_default(self):
        """Testet die Initialisierung mit Standardwerten (0, 0)."""
        asc = AffectiveStateCore()
        state = asc.get_state()
        self.assertEqual(state['x'], 0.0)
        self.assertEqual(state['y'], 0.0)

    def test_initialization_custom(self):
        """Testet die Initialisierung mit benutzerdefinierten Werten."""
        asc = AffectiveStateCore(initial_x=50, initial_y=-25)
        state = asc.get_state()
        self.assertEqual(state['x'], 50.0)
        self.assertEqual(state['y'], -25.0)

    def test_initialization_clamps_values(self):
        """Testet, ob die Initialisierung Werte außerhalb der Grenzen klemmt."""
        asc = AffectiveStateCore(initial_x=200, initial_y=-150)
        state = asc.get_state()
        self.assertEqual(state['x'], 100.0)
        self.assertEqual(state['y'], -100.0)

    def test_get_state(self):
        """Testet das Format des von get_state() zurückgegebenen Dictionarys."""
        asc = AffectiveStateCore(initial_x=10, initial_y=20)
        self.assertDictEqual(asc.get_state(), {'x': 10.0, 'y': 20.0})

    def test_update_state_positive(self):
        """Testet eine einfache positive Zustandsaktualisierung."""
        asc = AffectiveStateCore()
        asc.update_state(delta_x=10, delta_y=15.5)
        state = asc.get_state()
        self.assertEqual(state['x'], 10.0)
        self.assertEqual(state['y'], 15.5)

    def test_update_state_clamps_at_positive_100(self):
        """Testet, ob update_state() korrekt am oberen Limit (+100) klemmt."""
        asc = AffectiveStateCore(initial_x=95, initial_y=98)
        asc.update_state(delta_x=10, delta_y=10)
        state = asc.get_state()
        self.assertEqual(state['x'], 100.0)
        self.assertEqual(state['y'], 100.0)

    def test_update_state_clamps_at_negative_100(self):
        """Testet, ob update_state() korrekt am unteren Limit (-100) klemmt."""
        asc = AffectiveStateCore(initial_x=-95, initial_y=-98)
        asc.update_state(delta_x=-10, delta_y=-10)
        state = asc.get_state()
        self.assertEqual(state['x'], -100.0)
        self.assertEqual(state['y'], -100.0)

    def test_set_state_normal(self):
        """Testet das Setzen von normalen, gültigen Werten."""
        asc = AffectiveStateCore()
        asc.set_state(x=-50, y=75)
        state = asc.get_state()
        self.assertEqual(state['x'], -50.0)
        self.assertEqual(state['y'], 75.0)

    def test_set_state_clamps_above_upper_bound(self):
        """Testet, ob set_state() Werte über +100 korrekt klemmt."""
        asc = AffectiveStateCore()
        asc.set_state(x=120, y=999)
        state = asc.get_state()
        self.assertEqual(state['x'], 100.0)
        self.assertEqual(state['y'], 100.0)

    def test_set_state_clamps_below_lower_bound(self):
        """Testet, ob set_state() Werte unter -100 korrekt klemmt."""
        asc = AffectiveStateCore()
        asc.set_state(x=-150, y=-2000)
        state = asc.get_state()
        self.assertEqual(state['x'], -100.0)
        self.assertEqual(state['y'], -100.0)

    def test_state_remains_clamped_after_multiple_updates(self):
        """Stellt sicher, dass der Zustand auch nach mehreren Updates geklemmt bleibt."""
        asc = AffectiveStateCore(initial_x=99.0)
        asc.update_state(delta_x=5.0) # Sollte auf 100 klemmen
        self.assertEqual(asc.get_state()['x'], 100.0)
        asc.update_state(delta_x=5.0) # Sollte bei 100 bleiben
        self.assertEqual(asc.get_state()['x'], 100.0)

if __name__ == '__main__':
    unittest.main()