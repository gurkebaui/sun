# arena.py
import torch
import sys
import argparse

# Stellen Sie sicher, dass die Haupt-Module importiert werden können
sys.path.append('.')

from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore

# --- Modell-Hyperparameter (nur für die Initialisierung benötigt) ---
VOCAB_SIZE = 50
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512


def print_help():
    """Zeigt eine formatierte Liste aller verfügbaren Befehle an."""
    print("\n" + "=" * 25 + " CAPA Arena Befehle " + "=" * 25)
    print("  status                  : Zeigt den aktuellen Zustand des Agenten an.")
    print("  tick <n>                : Simuliert <n> Zeitschritte für interne Systeme.")
    print("  add_mem <text>          : Fügt eine manuelle Erinnerung hinzu.")
    print("  infer [--verbose] <text>: Startet den Gedankenprozess mit einem Text-Input.")
    print("  reward <wert>           : Erhöht die Valenz (y) um einen Wert.")
    print("  punish <wert>           : Verringert die Valenz (y) um einen Wert.")
    print("  stress <wert>           : Erhöht das Arousal (x) um einen Wert.")
    print("  calm <wert>             : Verringert das Arousal (x) um einen Wert.")
    print("  help                    : Zeigt diese Hilfe an.")
    print("  exit                    : Beendet die Arena.")
    print("=" * 72 + "\n")


def main():
    """
    Die Hauptfunktion der Arena. Initialisiert den Agenten und startet die
    interaktive Befehlsschleife mit argparse.
    """
    print("Initialisiere CAPA Agent...")
    pag = PAG_Model()
    asc = AffectiveStateCore()
    agent = CAPA_Agent(pag_model=pag, asc=asc)

    # Argument-Parser für eine saubere Befehlsverarbeitung
    parser = argparse.ArgumentParser(description="CAPA Arena CLI", add_help=False)
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    # Befehl: status
    subparsers.add_parser("status", help="Zeigt den aktuellen Zustand des Agenten an.")

    # Befehl: tick
    tick_parser = subparsers.add_parser("tick", help="Simuliert n Zeitschritte.")
    tick_parser.add_argument("num_ticks", type=int, help="Die Anzahl der zu simulierenden Ticks.")

    # Befehl: reward
    reward_parser = subparsers.add_parser("reward", help="Erhöht die Valenz (y).")
    reward_parser.add_argument("value", type=float, help="Der Wert, um den die Valenz erhöht wird.")

    # Befehl: punish
    punish_parser = subparsers.add_parser("punish", help="Verringert die Valenz (y).")
    punish_parser.add_argument("value", type=float, help="Der Wert, um den die Valenz verringert wird.")

    # Befehl: stress
    stress_parser = subparsers.add_parser("stress", help="Erhöht das Arousal (x).")
    stress_parser.add_argument("value", type=float, help="Der Wert, um den das Arousal erhöht wird.")

    # Befehl: calm
    calm_parser = subparsers.add_parser("calm", help="Verringert das Arousal (x).")
    calm_parser.add_argument("value", type=float, help="Der Wert, um den das Arousal verringert wird.")

    # Befehl: add_mem
    mem_parser = subparsers.add_parser("add_mem", help="Fügt eine manuelle Erinnerung hinzu.")
    mem_parser.add_argument("text", nargs='+', help="Der Text der Erinnerung.")

    # Befehl: infer
    infer_parser = subparsers.add_parser("infer", help="Startet den bewussten Gedankenprozess.")
    infer_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Zeigt den vollständigen Prompt vor der Antwort an.")
    infer_parser.add_argument("text", nargs='+', help="Der Text, der die aktuelle Situation beschreibt.")

    # Befehl: help & exit
    subparsers.add_parser("help", help="Zeigt die Hilfe an.")
    subparsers.add_parser("exit", help="Beendet die Anwendung.")

    print("CAPA Arena initialisiert. Agent ist bereit.")
    print("Geben Sie 'help' ein, um eine Liste der Befehle zu sehen.")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            try:
                # Parse die Argumente aus dem User-Input
                args = parser.parse_args(user_input.split())
            except SystemExit:
                # argparse beendet standardmäßig das Programm bei Fehlern oder -h.
                # Wir fangen das ab, damit die Schleife weiterläuft.
                continue

            if args.command == "exit":
                print("Fahre die Arena herunter. Auf Wiedersehen.")
                break

            elif args.command == "help":
                print_help()

            elif args.command == "status":
                state = agent.asc.get_state()
                pressure = agent.swhor.sleep_pressure
                sleeping_status = "Schlafend" if agent.swhor.is_sleeping else "Wach"
                print(f"ASC State: {{'x': {state['x']:.2f}, 'y': {state['y']:.2f}}}")
                print(f"SWHoR Status: Schlafdruck: {pressure:.2f} | Zustand: {sleeping_status}")
                print(f"Memory Status: {agent.memory.get_memory_count()} Erinnerungen gespeichert.")

            elif args.command == "tick":
                for _ in range(args.num_ticks):
                    agent.update()
                print(f"{args.num_ticks} Ticks simuliert.")

            elif args.command == "reward":
                agent.asc.update_state(delta_y=args.value)
                print(f"Valenz (y) um {args.value:.2f} erhöht.")

            elif args.command == "punish":
                agent.asc.update_state(delta_y=-args.value)
                print(f"Valenz (y) um {args.value:.2f} verringert.")

            elif args.command == "stress":
                agent.asc.update_state(delta_x=args.value)
                print(f"Arousal (x) um {args.value:.2f} erhöht.")

            elif args.command == "calm":
                agent.asc.update_state(delta_x=-args.value)
                print(f"Arousal (x) um {args.value:.2f} verringert.")

            elif args.command == "add_mem":
                text = " ".join(args.text)
                current_state = agent.asc.get_state()
                agent.memory.add_experience(text, metadata=current_state)

            elif args.command == "infer":
                situation_text = " ".join(args.text)
                final_answer, full_prompt = agent.run_inference_cycle(situation_text)

                if args.verbose:
                    print("\n" + "=" * 20 + " DEBUG: VOLLSTÄNDIGER PROMPT " + "=" * 20)
                    print(full_prompt)
                    print("=" * 66 + "\n")

                print(f"Agent > {final_answer}")

        except Exception as e:
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


if __name__ == '__main__':
    main()