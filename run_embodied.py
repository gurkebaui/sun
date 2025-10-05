# run_embodied.py
import time
import argparse
from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
import threading
import queue

# Globale Queue, um die Eingabe aus dem Thread zu erhalten
input_queue = queue.Queue()


def non_blocking_input(q):
    """Läuft in einem separaten Thread und wartet auf Benutzereingaben."""
    while True:
        user_input = input()
        q.put(user_input)


def main():
    parser = argparse.ArgumentParser(description="Startet den verkörperten CAPA-Agenten.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Zeigt die vollständigen Prompts an.")
    parser.add_argument("--cycle_time", type=int, default=5, help="Zeit in Sekunden zwischen den Bewusstseins-Zyklen.")
    args = parser.parse_args()

    print("--- SYSTEMSTART: VERKÖRPERTER MODUS ---")

    pag = PAG_Model(model_name="google/flan-t5-xl")
    asc = AffectiveStateCore()
    agent = CAPA_Agent(pag_model=pag, asc=asc)

    print("\n--- AGENT IST AKTIV. BEGINNE BEWUSSTSEINS-SCHLEIFE. (Beenden mit Strg+C) ---")
    print(
        "Geben Sie Text ein und drücken Sie Enter, um die Audio-Wahrnehmung für den nächsten Zyklus zu überschreiben.")

    # Starte den non-blocking Input Thread
    input_thread = threading.Thread(target=non_blocking_input, args=(input_queue,), daemon=True)
    input_thread.start()

    try:
        while True:
            print("\n" + "-" * 70)
            print("Neuer Zyklus... Warten auf Input oder fahre mit Wahrnehmung fort.")

            sensory_report = ""
            user_text_override = None

            # Prüfe, ob eine Benutzereingabe vorliegt
            try:
                user_text_override = input_queue.get_nowait()
            except queue.Empty:
                pass  # Keine Eingabe, fahre normal fort

            if user_text_override:
                print(f"Manuelle Eingabe empfangen: '{user_text_override}'")
                # Simuliere einen Sinneseindruck mit dem Text des Nutzers
                vision_report = agent.perception._perceive_vision()  # Nimm trotzdem wahr, was du siehst
                sound_report = "I hear ambient sounds like: Speech."
                speech_report = f"I hear someone say: '{user_text_override}'"
                sensory_report = f"Sensory Input:\n- {vision_report}\n- {sound_report}\n- {speech_report}"
                print(sensory_report)
            else:
                # Führe den normalen Wahrnehmungszyklus aus
                sensory_report = agent.perception.perceive()

            # Der restliche Zyklus bleibt gleich
            final_answer, full_prompt = agent.run_inference_cycle(sensory_report)

            if args.verbose:
                print("\n" + "=" * 20 + " DEBUG: VOLLSTÄNDIGER PROMPT " + "=" * 20)
                print(full_prompt)
                print("=" * 66 + "\n")

            print(f"Agent > {final_answer}")

            agent._run_background_processes()  # Führe die internen Systeme aus

            time.sleep(args.cycle_time)

    except KeyboardInterrupt:
        print("\n\n--- SYSTEM WIRD HERUNTERGEFAHREN ---")
        if agent.memory:
            agent.memory.shutdown()
        print("System erfolgreich heruntergefahren.")


if __name__ == '__main__':
    main()