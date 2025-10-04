# agent.py
import torch
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
from modulation.functions import (
    modulate_temperature,
    modulate_attention_scores,
    calculate_layer_drop_rate
)

from swhor.regulator import SWHoR
from vigilance.subsystem import VigilanceSubsystem


class CAPA_Agent:
    """
       Der Haupt-Controller, der alle Kernkomponenten (PAG, ASC, Modulation)
       integriert und steuert.
       """

    def __init__(self, pag_model: PAG_Model, asc: AffectiveStateCore):
        self.pag = pag_model
        self.asc = asc
        self.swhor = SWHoR()
        self.vigilance = VigilanceSubsystem()

        # NEU: Haupt-Update-Schleife des Agenten

    def update(self):
        current_state = self.asc.get_state()
        # --- KORRIGIERTE PSSC-LOGIK ---
        # 1. Prüfen, ob der Agent schlafen will (hoher Schlafdruck)
        wants_to_sleep = self.swhor.sleep_pressure > self.swhor.PRESSURE_THRESHOLD

        # 2. Wenn er schlafen will, aber noch nicht schläft, den Sicherheitscheck durchführen
        if wants_to_sleep and not self.swhor.is_sleeping:
            is_safe = self.vigilance.is_safe_to_sleep(current_state['x'])
            if is_safe:
                # Es ist sicher -> Schlaf einleiten, indem x negativ gesetzt wird.
                # Wir setzen x auf einen moderaten negativen Wert.
                self.asc.set_state(x=-50, y=current_state['y'])
                print("[Agent] Hoher Schlafdruck und sichere Umgebung. Leite Schlaf ein.")
            else:
                # Es ist nicht sicher -> wach bleiben und weiter leiden.
                print("[Agent] Hoher Schlafdruck, aber Umgebung unsicher. Schlaf wird verhindert.")

        # 3. Den SWHoR normal updaten lassen, der sich mit dem neuen Zustand synchronisiert.
        # Der Zustand des ASC wird hier erneut abgerufen, falls er sich geändert hat.
        swhor_deltas = self.swhor.update(self.asc.get_state()['x'])
        self.asc.update_state(
            delta_x=swhor_deltas['delta_x'],
            delta_y=swhor_deltas['delta_y']
        )

    def handle_stimulus(self, stimulus: dict):
        """
        NEU: Verarbeitet einen externen Reiz und löst bei Bedarf
        ein Not-Aufwachen aus.
        """
        print(f"Agent nimmt Reiz wahr: Intensität {stimulus.get('intensity', 0)}")
        if self.swhor.is_sleeping:
            is_danger = self.vigilance.filter_stimulus(stimulus)
            if is_danger:
                print("!!! ALARM: GEFAHR ERKANNT !!!")
                # 1. Schlafprozess sofort abbrechen
                self.swhor.interrupt_sleep()
                # 2. ASC in posttraumatischen Zustand versetzen
                self.asc.set_state(x=90, y=-80)
                print("Agent in Alarmbereitschaft versetzt.")

    def run_inference(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Führt eine Inferenz aus, die durch den aktuellen Zustand des ASC
        moduliert wird.

        Args:
            input_sequence (torch.Tensor): Der Eingabe-Tensor für den PAG.

        Returns:
            torch.Tensor: Der Ausgabe-Tensor vom PAG.
        """
        # 1. Aktuellen Zustand vom ASC abrufen
        current_state = self.asc.get_state()
        x, y = current_state['x'], current_state['y']

        # 2. Modulationsfunktionen aufrufen, um dynamische Hyperparameter zu berechnen
        temp = modulate_temperature(x, y)  # Übergebe jetzt x und y
        attn_func = lambda scores: modulate_attention_scores(x, scores)
        drop_rate = calculate_layer_drop_rate(x)

        print(f"Running inference with ASC state [x={x:.2f}, y={y:.2f}] -> "
              f"temp={temp:.2f}, drop_rate={drop_rate:.2f}")

        # 3. PAG-Inferenz aufrufen
        output = self.pag.infer(
            input_sequence,
            temperature=temp,
            attention_mod_func=attn_func,
            layer_drop_rate=drop_rate
        )

        return output