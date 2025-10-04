# vigilance/subsystem.py

class VigilanceSubsystem:
    """
    Kapselt die "Überlebensinstinkte" des Agenten.
    Besteht aus dem Pre-Sleep Safety Check (PSSC) und dem
    Sleep-Vigilance-Filter (SVF).
    """
    PSSC_AROUSAL_THRESHOLD = 20.0
    SVF_INTENSITY_THRESHOLD = 80.0

    def is_safe_to_sleep(self, asc_x: float) -> bool:
        """
        PSSC: Prüft, ob die Umgebung und der interne Zustand sicher genug
        sind, um einzuschlafen.

        Args:
            asc_x (float): Der aktuelle Arousal-Wert des Agenten.

        Returns:
            bool: True, wenn der Agent ruhig genug ist, um zu schlafen.
        """
        return asc_x <= self.PSSC_AROUSAL_THRESHOLD

    def filter_stimulus(self, stimulus: dict) -> bool:
        """
        SVF: Filtert einen externen Reiz und entscheidet, ob er
        gefährlich genug ist, um ein Not-Aufwachen auszulösen.

        Args:
            stimulus (dict): Ein Dictionary, das den Reiz beschreibt,
                             z.B. {'type': 'sound', 'intensity': 95}.

        Returns:
            bool: True (Alarm), wenn der Reiz den Schwellenwert überschreitet.
        """
        intensity = stimulus.get('intensity', 0.0)
        return intensity > self.SVF_INTENSITY_THRESHOLD
