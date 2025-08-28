# agent.py
import torch
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
from modulation.functions import (
    modulate_temperature,
    modulate_attention_scores,
    calculate_layer_drop_rate
)


class CAPA_Agent:
    """
    Der Haupt-Controller, der alle Kernkomponenten (PAG, ASC, Modulation)
    integriert und steuert.
    """

    def __init__(self, pag_model: PAG_Model, asc: AffectiveStateCore):
        """
        Initialisiert den Agenten mit den erforderlichen Modulen.

        Args:
            pag_model (PAG_Model): Das trainierte "Gehirn" des Agenten.
            asc (AffectiveStateCore): Die "emotionale" Steuereinheit.
        """
        self.pag = pag_model
        self.asc = asc

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