# modulation/functions.py
import random
from typing import List

def modulate_temperature(x_value: float, y_value: float) -> float:
    """
    Übersetzt Valenz (y) und Arousal (x) in einen Temperatur-Wert.

    - Positive Valenz (y > 0) erhöht die Temperatur (kreativer, explorativer).
    - Negative Valenz (y < 0) senkt die Temperatur (vorsichtiger, präziser).
    - Hohes Arousal (x > 50) senkt die Temperatur zusätzlich (fokussierter).

    Args:
        x_value (float): Der Arousal-Wert aus dem ASC, im Bereich [-100, 100].
        y_value (float): Der Valenz-Wert aus dem ASC, im Bereich [-100, 100].

    Returns:
        float: Der berechnete Temperatur-Wert, garantiert >= 0.01.
    """
    # 1. Basis-Temperatur basierend auf Valenz (y) berechnen - LOGIK INVERTIERT
    # Formel: 1.0 + (y_value / 100.0)
    # Bei y=100 (max Belohnung) -> temp=2.0 (max Kreativität)
    # Bei y=-100 (max Bestrafung) -> temp=0.0 (max Vorsicht)
    base_temp = 1.0 + (y_value / 100.0)

    # 2. Temperatur basierend auf Arousal (x) weiter modifizieren (diese Logik bleibt gleich)
    final_temp = base_temp
    if x_value >= 100:
        final_temp = 0.3 # Harter Override
    elif x_value > 50:
        reduction_factor = (x_value - 50.0) / 50.0
        temp_range = base_temp - 0.3
        reduction_amount = temp_range * reduction_factor
        final_temp = base_temp - reduction_amount

    # 3. Finalen Sicherheits-Clamp anwenden
    return max(0.01, final_temp)

def modulate_attention_scores(x_value: float, scores: List[float]) -> List[float]:
    """
    Manipuliert eine Liste von Attention Scores basierend auf dem Arousal-Wert (x).

    Hoher Fokus schärft die Attention, während niedriger Fokus/Verwirrung
    Rauschen hinzufügt, um die Attention zu "verwischen".

    Args:
        x_value (float): Der Arousal-Wert aus dem ASC, im Bereich [-100, 100].
        scores (List[float]): Eine Liste von float-Werten, die die
                              Attention Scores repräsentieren.

    Returns:
        List[float]: Die modifizierte Liste der Attention Scores.
    """
    if x_value > 50:
        # "Schärfungs"-Effekt: Verstärkt die Unterschiede zwischen den Scores.
        # Faktor geht von 1.0 (bei x=50) bis 2.0 (bei x=100).
        sharpening_factor = 1.0 + ((x_value - 50.0) / 50.0)
        return [score * sharpening_factor for score in scores]
    elif x_value < 0:
        # "Rauschen"-Effekt: Fügt zufälliges Rauschen hinzu.
        # Die Stärke des Rauschens skaliert mit der Negativität von x.
        # Wir definieren eine maximale Rausch-Amplitude, z.B. 10% des Score-Ranges.
        max_noise_amplitude = (max(scores) - min(scores)) * 0.1 if scores else 0
        noise_strength = (abs(x_value) / 100.0) * max_noise_amplitude
        return [score + random.uniform(-noise_strength, noise_strength) for score in scores]
    else:
        # Im Bereich [0, 50] (normaler Fokus) bleiben die Scores unverändert.
        return scores

def calculate_layer_drop_rate(x_value: float) -> float:
    """
    Bestimmt die Rate der zu überspringenden Layer bei extremem Arousal.

    Dies simuliert einen "Tunnelblick" oder eine kognitive Abkürzung
    in Zuständen hoher Dringlichkeit (Alarmbereitschaft).

    Args:
        x_value (float): Der Arousal-Wert aus dem ASC, im Bereich [-100, 100].

    Returns:
        float: Die Layer-Drop-Rate, ein Wert zwischen 0.0 und 0.5.
    """
    if x_value < 90:
        return 0.0
    else:
        # Lineare Skalierung von 0.0 bei x=90 auf 0.5 bei x=100.
        rate = 0.5 * ((x_value - 90.0) / 10.0)
        # Clamp, um sicherzustellen, dass die Rate 0.5 nicht überschreitet,
        # falls x_value > 100 übergeben wird.
        return min(0.5, max(0.0, rate))