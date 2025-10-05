# run_embodied.py
import time
import argparse
from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore


def main():
    parser = argparse.ArgumentParser(description="Startet den verkörperten CAPA-Agenten.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Zeigt die vollständigen Prompts an.")
    parser.add_argument("--cycle_time", type=int, default=5, help="Zeit in Sekunden zwischen den Bewusstseins-Zyklen.")
    args = parser.parse_args()

    print("--- SYSTEMSTART: VERKÖRPERTER MODUS ---")

    # Initialisiere die Kernkomponenten
    pag = PAG_Model(model_name="google/flan-t5-base")
    asc = AffectiveStateCore()
    agent = CAPA_Agent(pag_model=pag, asc=asc)

    print("\n--- AGENT IST AKTIV. BEGINNE BEWUSSTSEINS-SCHLEIFE. (Beenden mit Strg+C) ---")

    try:
        while True:
            # Rufe die zentrale, in sich geschlossene Update-Methode des Agenten auf.
            agent.update(verbose=args.verbose)

            # Wartezeit zwischen den Zyklen
            print("-" * 70)
            time.sleep(args.cycle_time)

    except KeyboardInterrupt:
        print("\n\n--- SYSTEM WIRD HERUNTERGEFAHREN ---")
        if agent.memory:
            agent.memory.shutdown()
        print("System erfolgreich heruntergefahren.")


if __name__ == '__main__':
    main()