# -*- coding: utf-8 -*-
# asc/core.py

class AffectiveStateCore:
    """
    Verwaltet und kapselt den zentralen [x,y]-Zustandsvektor von CAPA.

    Diese Klasse ist die alleinige Wahrheitsquelle für den "mentalen" Zustand
    des Agenten, repräsentiert durch Arousal/Fokus (x) und Valenz/Gefühl (y).
    Sie stellt sicher, dass die Zustandswerte immer innerhalb der
    definierten Grenzen von -100 bis +100 bleiben.
    """

    MIN_VALUE = -100.0
    MAX_VALUE = 100.0

    def __init__(self, initial_x: float = 0.0, initial_y: float = 0.0):
        """
        Initialisiert den Affective State Core.

        Args:
            initial_x (float): Der anfängliche Wert für Arousal/Fokus.
                               Standardmäßig 0.0 (ruhig).
            initial_y (float): Der anfängliche Wert für Valenz/Gefühl.
                               Standardmäßig 0.0 (neutral).
        """
        self.x = self._clamp(initial_x)
        self.y = self._clamp(initial_y)

    def _clamp(self, value: float) -> float:
        """
        Eine private Hilfsmethode, um einen Wert innerhalb der
        MIN_VALUE und MAX_VALUE Grenzen zu halten.
        """
        return max(self.MIN_VALUE, min(value, self.MAX_VALUE))

    def get_state(self) -> dict:
        """
        Gibt den aktuellen Zustand als Dictionary zurück.

        Returns:
            dict: Ein Dictionary mit den aktuellen x- und y-Werten,
                  z.B. {'x': 10.5, 'y': -25.0}.
        """
        return {'x': self.x, 'y': self.y}

    def update_state(self, delta_x: float = 0.0, delta_y: float = 0.0):
        """
        Modifiziert den aktuellen Zustand durch relative Änderungen (Deltas).

        Die resultierenden Werte werden automatisch auf den Bereich
        [-100, 100] geklemmt.

        Args:
            delta_x (float): Die Änderung, die auf den x-Wert angewendet wird.
            delta_y (float): Die Änderung, die auf den y-Wert angewendet wird.
        """
        self.x = self._clamp(self.x + delta_x)
        self.y = self._clamp(self.y + delta_y)

    def set_state(self, x: float, y: float):
        """
        Setzt den Zustand auf absolute Werte.

        Die Werte werden automatisch auf den Bereich [-100, 100] geklemmt,
        bevor sie zugewiesen werden.

        Args:
            x (float): Der neue absolute Wert für die x-Achse.
            y (float): Der neue absolute Wert für die y-Achse.
        """
        self.x = self._clamp(x)
        self.y = self._clamp(y)