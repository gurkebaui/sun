# pag/infer.py
import torch
from pag.model import PAG_Model

# --- Dieselben Hyperparameter wie beim Training verwenden ---
VOCAB_SIZE = 50
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Spezielle Tokens
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


def main():
    """Hauptfunktion für die Inferenz."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    # Modellarchitektur instanziieren
    model = PAG_Model(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
    ).to(device)

    # Trainierte Gewichte laden
    try:
        model.load_state_dict(torch.load("pag/pag_model.pth", map_location=device))
    except FileNotFoundError:
        print("Fehler: Modelldatei 'pag/pag_model.pth' nicht gefunden.")
        print("Bitte führen Sie zuerst 'train.py' aus, um das Modell zu trainieren.")
        return

    model.eval()

    # --- Testfall ---
    # Eingabesequenz: 4 8 15 16 23 42
    test_sequence = [4, 8, 15, 16, 24, 42, 34, 43]

    # In Tensor umwandeln und SOS/EOS hinzufügen
    src_tensor = torch.tensor([SOS_TOKEN] + test_sequence + [EOS_TOKEN]).to(device)

    print("-" * 50)
    print(f"Eingabesequenz: {test_sequence}")

    # Inferenz mit Standardparametern durchführen
    with torch.no_grad():
        prediction = model.infer(src_tensor)

    # Ergebnis dekodieren (SOS entfernen, bei EOS stoppen)
    output_sequence = [
        item.item() for item in prediction[1:] if item.item() != EOS_TOKEN
    ]

    print(f"Vorhergesagte Umkehrung: {output_sequence}")
    print(f"Erwartete Umkehrung:   {test_sequence[::-1]}")
    print("-" * 50)

    # Demonstration der Modulationsparameter (Ticket [PAG-HOOKS])
    print("Demonstration der Inferenz-Schnittstelle mit Modulationsparametern:")
    with torch.no_grad():
        # Der Aufruf funktioniert, auch wenn die Logik dahinter noch nicht voll implementiert ist
        prediction_mod = model.infer(
            src_tensor,
            temperature=1.5,
            attention_mod_func=lambda x: x,  # Dummy-Funktion
            layer_drop_rate=0.1
        )
    output_sequence_mod = [
        item.item() for item in prediction_mod[1:] if item.item() != EOS_TOKEN
    ]
    print(f"Vorhersage mit temp=0.5: {output_sequence_mod} (sollte identisch sein, da greedy)")


if __name__ == '__main__':
    main()