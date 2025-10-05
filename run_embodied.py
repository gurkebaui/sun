# run_embodied.py
import time
import argparse
from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
import keyboard


class InputHandler:
    """Eine robuste Klasse zur Verarbeitung von Tastatureingaben."""

    def __init__(self):
        self.buffer = ""
        self.enter_pressed = False
        keyboard.on_press(self.on_key_press)

    def on_key_press(self, event):
        if event.name == 'enter':
            self.enter_pressed = True
        elif event.name == 'backspace':
            self.buffer = self.buffer[:-1]
        elif len(event.name) == 1:  # Nur druckbare Zeichen hinzufügen
            self.buffer += event.name

        # Zeige dem Nutzer, was er tippt
        print(f"\rEingabe: {self.buffer}", end="")

    def get_input(self):
        if self.enter_pressed:
            captured_input = self.buffer
            self.buffer = ""
            self.enter_pressed = False
            print()  # Neue Zeile nach der Eingabe
            return captured_input
        return None


def main():
    parser = argparse.ArgumentParser(description="Startet den verkörperten CAPA-Agenten.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Zeigt die vollständigen Prompts an.")
    parser.add_argument("--cycle_time", type=int, default=5, help="Zeit in Sekunden zwischen den Bewusstseins-Zyklen.")
    args = parser.parse_args()

    print("--- SYSTEMSTART: VERKÖRPERTER MODUS ---")

    pag = PAG_Model(model_name="google/flan-t5-xl")
    asc = AffectiveStateCore()
    agent = CAPA_Agent(pag_model=pag, asc=asc)
    input_handler = InputHandler()

    print("Richte Echtzeit-Feedback-Hotkeys ein (+, -, *, /)...")
    keyboard.add_hotkey('+', lambda: (agent.asc.update_state(delta_y=10), print("\n[Feedback: +10y]")))
    keyboard.add_hotkey('-', lambda: (agent.asc.update_state(delta_y=-10), print("\n[Feedback: -10y]")))
    keyboard.add_hotkey('*', lambda: (agent.asc.update_state(delta_x=10), print("\n[Feedback: +10x]")))
    keyboard.add_hotkey('/', lambda: (agent.asc.update_state(delta_x=-10), print("\n[Feedback: -10x]")))

    print("\n--- AGENT IST AKTIV. BEGINNE BEWUSSTSEINS-SCHLEIFE. (Beenden mit Strg+C) ---")
    print("Tippen Sie, um die Audio-Wahrnehmung zu überschreiben. Drücken Sie Enter, um zu senden.")

    try:
        while True:
            agent.update(verbose=args.verbose, text_override=input_handler.get_input())
            print("-" * 70)
            time.sleep(args.cycle_time)

    except KeyboardInterrupt:
        print("\n\n--- SYSTEM WIRD HERUNTERGEFAHREN ---")
        keyboard.unhook_all()
        if agent.memory:
            agent.memory.shutdown()
        print("System erfolgreich heruntergefahren.")


if __name__ == '__main__':
    main()