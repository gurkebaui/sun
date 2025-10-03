# swhor/regulator.py

class SWHoR:
    """
    Sleep-Wake Homeostasis Regulator (SWHoR).
    Verwaltet den Schlafzyklus, den Schlafdruck und die damit verbundenen
    Belohnungen und Bestrafungen des CAPA-Agenten.
    """
    # Konfigurationskonstanten
    PRESSURE_THRESHOLD = 80.0
    MAX_PRESSURE = 150.0
    PRESSURE_BUILD_BASE_RATE = 0.5
    FATIGUE_PENALTY = -0.1
    SLEEP_RECOVERY_RATE = 1.0
    SLEEP_REWARD = 0.2

    # Konstanten für das optimale Schlaffenster
    OPTIMAL_SLEEP_FACTOR = 0.8  # Schlafdauer = Druck * Faktor
    WAKE_REWARD_FACTOR = 10.0
    OVERSLEEP_PENALTY_FACTOR = -15.0

    def __init__(self):
        """Initialisiert den SWHoR im wachen, ausgeruhten Zustand."""
        self.sleep_pressure: float = 0.0
        self.is_sleeping: bool = False
        self.current_sleep_duration: int = 0
        self.pressure_at_sleep_start: float = 0.0

    def update(self, asc_x: float) -> dict:
        """
        Führt einen Simulationsschritt (tick) aus.
        Diese Version ist atomar und behandelt Zustandswechsel robust.
        """
        deltas = {'delta_x': 0.0, 'delta_y': 0.0}
        agent_is_currently_in_sleep_state = asc_x < 0

        # --- 1. Zustands-Synchronisation ---
        # Prüft auf Diskrepanzen zwischen dem ASC-Zustand und dem internen SWHoR-Zustand.

        # Fall A: Agent wurde manuell schlafen gelegt
        if not self.is_sleeping and agent_is_currently_in_sleep_state:
            self._on_sleep_start()

        # Fall B: Agent wurde manuell aufgeweckt
        elif self.is_sleeping and not agent_is_currently_in_sleep_state:
            deltas.update(self._calculate_wake_bonus())
            self.is_sleeping = False

        # --- 2. Hauptlogik basierend auf dem (jetzt korrekten) Zustand ---
        if self.is_sleeping:
            # --- SCHLAF-LOGIK ---
            self.sleep_pressure = max(0, self.sleep_pressure - self.SLEEP_RECOVERY_RATE)
            self.current_sleep_duration += 1
            deltas['delta_y'] += self.SLEEP_REWARD

            # Automatisches Aufwachen, wenn der Schlafdruck auf 0 sinkt
            if self.sleep_pressure <= 0:
                wake_bonus = self._calculate_wake_bonus()
                deltas['delta_y'] += wake_bonus.get('delta_y', 0.0)
                deltas['delta_x'] += abs(asc_x)  # Aufwachimpuls
                self.is_sleeping = False
        else:
            # --- WACH-LOGIK ---
            build_rate = self.PRESSURE_BUILD_BASE_RATE * (1 + asc_x / 100.0)
            self.sleep_pressure = min(self.MAX_PRESSURE, self.sleep_pressure + build_rate)

            if self.sleep_pressure > self.PRESSURE_THRESHOLD:
                deltas['delta_y'] += self.FATIGUE_PENALTY

        return deltas

    def _on_sleep_start(self):
        """Wird aufgerufen, wenn der Schlaf-Zustand beginnt."""
        self.is_sleeping = True
        self.current_sleep_duration = 0
        self.pressure_at_sleep_start = self.sleep_pressure
        print(f"[SWHoR] Schlaf eingeleitet. Schlafdruck bei Start: {self.pressure_at_sleep_start:.1f}")

    def _calculate_wake_bonus(self) -> dict:
        """Berechnet die Belohnung/Bestrafung beim Aufwachen."""
        optimal_duration = self.pressure_at_sleep_start * self.OPTIMAL_SLEEP_FACTOR
        duration_delta = self.current_sleep_duration - optimal_duration

        if self.pressure_at_sleep_start <= 0:
            print(f"[SWHoR] Aufgewacht. Kein Schlafbedürfnis vorhanden.")
            return {'delta_y': 0.0}

        if self.current_sleep_duration < optimal_duration:
            reward = (self.current_sleep_duration / optimal_duration) * self.WAKE_REWARD_FACTOR
            print(f"[SWHoR] Aufgewacht. Schlaf war erholsam. Belohnung: {reward:.1f}")
            return {'delta_y': reward}
        else:
            oversleep_ticks = max(0, duration_delta - (optimal_duration * 0.1))
            final_penalty = oversleep_ticks * self.OVERSLEEP_PENALTY_FACTOR
            print(f"[SWHoR] Aufgewacht. Zu lang geschlafen! Strafe: {final_penalty:.1f}")
            return {'delta_y': final_penalty}