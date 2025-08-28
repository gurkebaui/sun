# arena.py
import torch
import sys

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu, um lokale Module zu finden
# Dies ist nicht immer notwendig, stellt aber die Kompatibilität sicher.
sys.path.append('.')

from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore

# --- Modell-Hyperparameter (müssen mit dem Training übereinstimmen) ---
# Diese werden benötigt, um eine Instanz des PAG_Model zu erstellen.
VOCAB_SIZE = 50
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512


def print_help():
    """Zeigt eine formatierte Liste aller verfügbaren Befehle an."""
    print("\n--- CAPA Arena Befehle ---")
    print("  status                  : Zeigt den aktuellen [x,y] Zustand des ASC an.")
    print("  infer <seq>             : Führt eine Inferenz mit einer Zahlenfolge aus (z.B. infer 1 2 3).")
    print("  reward <wert>           : Erhöht die Valenz (y) um einen Wert (Belohnung).")
    print("  punish <wert>           : Verringert die Valenz (y) um einen Wert (Bestrafung).")
    print("  stress <wert>           : Erhöht das Arousal (x) um einen Wert (Stress/Fokus).")
    print("  calm <wert>             : Verringert das Arousal (x) um einen Wert (Entspannung).")
    print("  help                    : Zeigt diese Hilfe an.")
    print("  exit                    : Beendet die Arena.")
    print("--------------------------\n")


def main():
    """
    Die Hauptfunktion der Arena. Initialisiert den Agenten und startet die
    interaktive Befehlsschleife.
    """
    print("Initialisiere CAPA Agent...")
    # 1. Instanziiere die Kernkomponenten
    pag = PAG_Model(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD)
    # HINWEIS: Das PAG-Modell ist untrainiert, aber das ist für die Demonstration
    # der Systemmechanik ausreichend.
    asc = AffectiveStateCore()
    agent = CAPA_Agent(pag_model=pag, asc=asc)

    print("CAPA Arena initialisiert. Agent ist bereit.")
    print("Geben Sie 'help' ein, um eine Liste der Befehle zu sehen.")

    # 2. Starte die Endlosschleife für Benutzereingaben
    while True:
        try:
            user_input = input("> ").strip().lower()
            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0]
            args = parts[1:]

            if command == "exit":
                print("Fahre die Arena herunter. Auf Wiedersehen.")
                break

            elif command == "help":
                print_help()

            elif command == "status":
                state = agent.asc.get_state()
                print(f"ASC State: {{'x': {state['x']:.2f}, 'y': {state['y']:.2f}}}")

            elif command == "infer":
                if not args:
                    print("Fehler: Bitte geben Sie eine Zahlenfolge an (z.B. infer 1 2 3).")
                    continue
                # Konvertiere die Eingabesequenz in Zahlen
                seq = [int(num) for num in args]
                # Füge SOS/EOS-Tokens hinzu und erstelle einen Tensor
                input_tensor = torch.tensor([agent.pag.sos_token] + seq + [agent.pag.eos_token])

                output_tensor = agent.run_inference(input_tensor)

                # Dekodiere das Ergebnis für die Anzeige
                output_seq = [
                    item.item() for item in output_tensor[1:] if item.item() != agent.pag.eos_token
                ]
                print(f"Agent > {' '.join(map(str, output_seq))}")

            elif command == "reward":
                if len(args) != 1:
                    print("Fehler: Bitte geben Sie genau einen Wert an (z.B. reward 25.5).")
                    continue
                value = float(args[0])
                agent.asc.update_state(delta_y=value)
                print(f"Valenz (y) um {value:.2f} erhöht.")

            elif command == "punish":
                if len(args) != 1:
                    print("Fehler: Bitte geben Sie genau einen Wert an (z.B. punish 30).")
                    continue
                value = float(args[0])
                agent.asc.update_state(delta_y=-value)  # Wende den Wert als negativ an
                print(f"Valenz (y) um {value:.2f} verringert.")

            elif command == "stress":
                if len(args) != 1:
                    print("Fehler: Bitte geben Sie genau einen Wert an (z.B. stress 20).")
                    continue
                value = float(args[0])
                agent.asc.update_state(delta_x=value)
                print(f"Arousal (x) um {value:.2f} erhöht.")


            elif command == "calm":
                if len(args) != 1:
                    print("Fehler: Bitte geben Sie genau einen Wert an (z.B. calm 15).")
                    continue
                value = float(args[0])
                agent.asc.update_state(delta_x=-value)  # Wende den Wert als negativ an
                print(f"Arousal (x) um {value:.2f} verringert.")

            else:
                print(f"Fehler: Unbekannter Befehl '{command}'. Geben Sie 'help' für eine Liste der Befehle ein.")

        except ValueError:
            print("Fehler: Ungültige Eingabe. Bitte geben Sie für Werte eine gültige Zahl an.")
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


if __name__ == '__main__':
    main()