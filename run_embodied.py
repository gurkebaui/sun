# run_embodied.py
import os
import time
import argparse
import sys
import keyboard

# Stellen Sie sicher, dass die Haupt-Module importiert werden können
sys.path.append('.')

from agent import CAPA_Agent
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
from memory.subsystem import MemorySubsystem


class InputHandler:
    """Eine robuste Klasse zur Verarbeitung von Tastatureingaben."""

    def __init__(self):
        self.buffer = ""
        self.enter_pressed = False
        keyboard.on_press(self.on_key_press, suppress=False)

    def on_key_press(self, event):
        if event.name == 'enter':
            self.enter_pressed = True
        elif event.name == 'backspace':
            self.buffer = self.buffer[:-1]
        elif len(event.name) == 1:
            self.buffer += event.name

        # Zeige dem Nutzer, was er tippt
        print(f"\rEingabe: {self.buffer}   ", end="")

    def get_input(self):
        if self.enter_pressed:
            captured_input = self.buffer
            self.buffer = ""
            self.enter_pressed = False
            print()
            return captured_input
        return None


def print_help():
    """Zeigt eine formatierte Liste aller verfügbaren Befehle an."""
    print("\n" + "=" * 25 + " CAPA Befehle " + "=" * 25)
    print("  status                  : Zeigt den aktuellen Zustand des Agenten an.")
    print("  tick <n>                : Simuliert <n> autonome Zyklen (ohne Wahrnehmung).")
    print("  add_mem <text>          : Fügt eine manuelle Erinnerung hinzu.")
    print("  <text>                  : Jede andere Eingabe wird als direkter Sprach-Input verarbeitet.")
    print("  buffer                  : Zeigt den Inhalt des Kurzzeitgedächtnis-Puffers an.")
    print("  view_mem --latest <n>   : Zeigt die <n> neuesten Erinnerungen an.")
    print("  help                    : Zeigt diese Hilfe an.")
    print("  exit                    : Beendet die Anwendung.")
    print("Hotkeys (jederzeit): +, -, *, / für direktes Feedback.")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Startet den verkörperten CAPA-Agenten.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Zeigt die vollständigen Prompts an.")
    parser.add_argument("--cycle_time", type=int, default=5, help="Zeit in Sekunden zwischen den autonomen Zyklen.")
    args = parser.parse_args()

    print("--- SYSTEMSTART: VERKÖRPERTER MODUS ---")

    # --- KORREKTUR BEGINNT HIER ---

    # 1. DEN PFAD ZUR DATENBANK EXPLIZIT UND ROBUST FESTLEGEN (Bleibt gleich)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "capa_memory_db")

    # 2. DIE SUBSYSTEME HIER ERSTELLEN (Bleibt gleich)
    pag_model = PAG_Model(model_name="google/flan-t5-xl")
    asc = AffectiveStateCore()
    memory_system = MemorySubs--ystem(db_path=db_path)  # <-- Unser korrektes Gedächtnis mit 5 Einträgen

    # 3. AGENT ERSTELLEN UND DANN DAS GEDÄCHTNIS ZUWEISEN (Die neue Logik)

    # Zuerst den Agenten so erstellen, wie er es erwartet (ohne Gedächtnis-Argument).
    # Er wird intern ein leeres Gedächtnis erstellen.
    agent = CAPA_Agent(
        pag_model=pag_model,
        asc=asc
    )

    # JETZT überschreiben wir sein leeres Gedächtnis mit unserer
    # Instanz, die die Daten aus der Datenbank geladen hat.
    agent.memory = memory_system

    # --- KORREKTUR ENDET HIER ---
    input_handler = InputHandler()


    print("Richte Echtzeit-Feedback-Hotkeys ein (+, -, *, /)...")
    keyboard.add_hotkey('+', lambda: (agent.asc.update_state(delta_y=10), print("\n[Feedback: +10y]")))
    keyboard.add_hotkey('-', lambda: (agent.asc.update_state(delta_y=-10), print("\n[Feedback: -10y]")))
    keyboard.add_hotkey('*', lambda: (agent.asc.update_state(delta_x=10), print("\n[Feedback: +10x]")))
    keyboard.add_hotkey('/', lambda: (agent.asc.update_state(delta_x=-10), print("\n[Feedback: -10x]")))

    print("\n--- AGENT IST AKTIV. (Beenden mit Strg+C) ---")
    print("Autonomer Modus aktiv. Tippen Sie einen Befehl (z.B. 'status') und Enter.")

    try:
        while True:
            command_input = input_handler.get_input()

            if command_input:
                parts = command_input.lower().split()
                cmd = parts[0]
                cmd_args = parts[1:]

                if cmd == "exit":
                    break
                elif cmd == "help":
                    print_help()
                elif cmd == "status":
                    state = agent.asc.get_state()
                    pressure = agent.swhor.sleep_pressure
                    sleeping_status = "Schlafend" if agent.swhor.is_sleeping else "Wach"
                    print(f"ASC State: {{'x': {state['x']:.2f}, 'y': {state['y']:.2f}}}")
                    print(f"SWHoR Status: Schlafdruck: {pressure:.2f} | Zustand: {sleeping_status}")
                    print(f"Memory Status: {agent.memory.get_memory_count()} Erinnerungen.")
                    print(f"Buffer Status: {len(agent.experience_buffer)} Erlebnisse.")
                elif cmd == "tick" and cmd_args:
                    for _ in range(int(cmd_args[0])):
                        agent._run_background_processes()
                    print(f"{cmd_args[0]} autonome Zyklen simuliert.")
                elif cmd == "add_mem" and cmd_args:
                    agent.memory.add_experience(" ".join(cmd_args), agent.asc.get_state())
                elif cmd == "buffer":
                    print(f"--- Kurzzeitgedächtnis ({len(agent.experience_buffer)} Einträge) ---")
                    for i, exp in enumerate(agent.experience_buffer): print(f"{i + 1}: {exp['text']}")
                elif cmd == "view_mem" and cmd_args and cmd_args[0] == "--latest":
                    count = int(cmd_args[1]) if len(cmd_args) > 1 else 1
                    memories = agent.memory.get_latest_memories(n_results=count)
                    for i, mem in enumerate(memories): print(f"Neueste-{i}: {mem['text']} | Meta: {mem['metadata']}")
                else:
                    # Jede andere Eingabe wird als Sprach-Input behandelt
                    agent.update(verbose=args.verbose, text_override=command_input)
            else:
                # Autonomer Zyklus, wenn keine Eingabe erfolgt
                agent.update(verbose=args.verbose)

            print("-" * 70)
            time.sleep(args.cycle_time)

    except KeyboardInterrupt:
        print("\n\n--- SYSTEM WIRD HERUNTERGEFAHREN ---")
        keyboard.unhook_all()
        if agent.memory:
            agent.memory.shutdown()
        print("System erfolgreich heruntergefahren.")

    finally:
        # Dieser finally-Block stellt sicher, dass das Herunterfahren immer versucht wird.
        keyboard.unhook_all()
        if hasattr(agent, 'memory') and agent.memory is not None:
            agent.memory.shutdown()
        print("System erfolgreich heruntergefahren.")


if __name__ == '__main__':
    main()