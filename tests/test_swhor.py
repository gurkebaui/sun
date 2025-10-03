# tests/test_swhor.py
import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swhor.regulator import SWHoR
from asc.core import AffectiveStateCore

class TestSWHoR(unittest.TestCase):

    def setUp(self):
        self.swhor = SWHoR()
        self.asc = AffectiveStateCore()

    def test_pressure_builds_when_awake(self):
        initial_pressure = self.swhor.sleep_pressure
        self.swhor.update(self.asc.get_state()['x'])
        self.assertGreater(self.swhor.sleep_pressure, initial_pressure)

    def test_pressure_builds_faster_with_high_arousal(self):
        swhor_calm = SWHoR()
        swhor_stressed = SWHoR()
        for _ in range(10):
            swhor_calm.update(0) # x=0
            swhor_stressed.update(80) # x=80
        self.assertGreater(swhor_stressed.sleep_pressure, swhor_calm.sleep_pressure)

    def test_fatigue_penalty_at_high_pressure(self):
        self.swhor.sleep_pressure = 90
        deltas = self.swhor.update(0)
        self.assertEqual(deltas['delta_y'], self.swhor.FATIGUE_PENALTY)

    def test_sleep_reduces_pressure_and_rewards_y(self):
        self.swhor.sleep_pressure = 50
        # Schlaf einleiten
        self.asc.set_state(x=-50, y=0)
        deltas = self.swhor.update(self.asc.get_state()['x'])
        self.assertLess(self.swhor.sleep_pressure, 50)
        self.assertEqual(deltas['delta_y'], self.swhor.SLEEP_REWARD)

if __name__ == '__main__':
    unittest.main()