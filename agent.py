# agent.py
import torch
import torch
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
from swhor.regulator import SWHoR
from vigilance.subsystem import VigilanceSubsystem
from memory.subsystem import MemorySubsystem
from modulation.functions import modulate_temperature
from typing import List, Dict, Any
from perception.subsystem import PerceptionSubsystem

class CAPA_Agent:
    def __init__(self, pag_model: PAG_Model, asc: AffectiveStateCore):
        self.pag = pag_model
        self.asc = asc
        self.swhor = SWHoR()
        self.vigilance = VigilanceSubsystem()
        self.memory = MemorySubsystem()
        self.experience_buffer: List[Dict[str, Any]] = []
        self.perception = PerceptionSubsystem()

    def _consolidate_and_synthesize_memories(self):
        # ... (Diese Methode bleibt exakt wie von Ihnen bereitgestellt)
        print("\n=== BEGINN DER SCHLAFPHASE: KONSOLIDIERUNG & SYNTHESE ===")
        if not self.experience_buffer:
            print("Keine neuen Erlebnisse zum Konsolidieren. Schlaf ist rein erholsam.")
            return
        # ... (Rest der Logik)

    def update(self, verbose: bool = False):
        """
        Die zentrale "Bewusstseins"-Schleife.
        Nimmt die Welt wahr, reagiert und führt Hintergrundprozesse aus.
        """
        # 1. Autonome, unbewusste Prozesse (Herzschlag, Müdigkeit)
        self._run_background_processes()

        # Nur wenn der Agent wach ist, nimmt er aktiv wahr und denkt.
        if not self.swhor.is_sleeping:
            # 2. Bewusste Wahrnehmung der Umgebung
            sensory_report = self.perception.perceive()

            # 3. Kognitiver Zyklus ("Gedankenprozess")
            final_answer, full_prompt = self.run_inference_cycle(sensory_report)

            if verbose:
                print("\n" + "=" * 20 + " DEBUG: VOLLSTÄNDIGER PROMPT " + "=" * 20)
                print(full_prompt)
                print("=" * 66 + "\n")

            print(f"Agent > {final_answer}")
        else:
            print("Agent schläft... zZz...")

    def _run_background_processes(self):
        """Führt die autonomen Subsysteme aus."""
        current_state = self.asc.get_state()
        wants_to_sleep = self.swhor.sleep_pressure > self.swhor.PRESSURE_THRESHOLD

        if wants_to_sleep and not self.swhor.is_sleeping:
            is_safe = self.vigilance.is_safe_to_sleep(current_state['x'])
            if is_safe:
                self.asc.set_state(x=-50, y=current_state['y'])
                self._consolidate_and_synthesize_memories()

        swhor_deltas = self.swhor.update(self.asc.get_state()['x'])
        self.asc.update_state(
            delta_x=swhor_deltas['delta_x'],
            delta_y=swhor_deltas['delta_y']
        )


    def _construct_internal_monologue_prompt(self, situation_text: str, memories: List[Dict[str, Any]],
                                             state: Dict[str, float]) -> str:
        context = (
            f"Context:\n- My internal state is: Arousal={state['x']:.0f}, Valence={state['y']:.0f}.\n"
            f"- The current situation is: '{situation_text}'\n"
            "- My relevant memories are:\n"
        )
        if memories:
            unique_memories = list({mem['text']: mem for mem in memories}.values())
            for mem in unique_memories:
                context += f"  - I remember that '{mem['text']}'\n"
        else:
            context += "  - None\n"
        task = "\nTask: Based on all the context, what is my key conclusion or plan? (Internal thought only)"
        return context + task

    def _construct_final_response_prompt(self, situation_text: str, internal_thought: str) -> str:
        # FINALE KORREKTUR: Füge eine explizite Anweisung für die Sprache hinzu.
        persona = (
            "You are CAPA. You are helpful, factual, and concise. "
            "You do not ask questions back to the user. "
            "Your goal is to give the most direct and useful answer possible. "
            "You must always respond in English."
        )

        context = (
            f"Your internal thought is: '{internal_thought}'.\n"
            f"The user's request (derived from sensory input) is: '{situation_text}'."
        )
        task = "\nTask: Based on your personality and your internal thought, what is your immediate, direct response?\nYour Response:"
        return f"{persona}\n\n{context}\n\n{task}"

    def run_inference_cycle(self, situation_text: str) -> (str, str):
        print("\n--- Beginn des Zwei-Stufen-Denkprozesses ---")
        current_state = self.asc.get_state()
        temp = modulate_temperature(current_state['x'], current_state['y'])

        print(f"[1&2] Rufe relevante Erinnerungen ab...")
        relevant_memories = self.memory.query_relevant_memories(situation_text, n_results=3)

        print("[3a] Formuliere internen Gedanken...")
        internal_prompt = self._construct_internal_monologue_prompt(situation_text, relevant_memories, current_state)
        internal_thought = self.pag.infer(prompt=internal_prompt, temperature=temp)

        print("[3b] Formuliere externe Antwort basierend auf dem Gedanken...")
        final_prompt = self._construct_final_response_prompt(situation_text, internal_thought)
        final_answer = self.pag.infer(prompt=final_prompt, temperature=temp)

        print("[5] Kognitiver Prozess abgeschlossen. Erlebnis wird im Puffer gespeichert.")
        experience_summary = f"In the situation '{situation_text}', I thought '{internal_thought}' and responded: '{final_answer}'"
        self.experience_buffer.append({'text': experience_summary, 'metadata': current_state})

        # KORREKTUR 2: Referenzfehler behoben
        return final_answer, final_prompt

    def handle_stimulus(self, stimulus: dict):
        if self.swhor.is_sleeping:
            is_danger = self.vigilance.filter_stimulus(stimulus)
            if is_danger:
                self.swhor.interrupt_sleep()
                self.asc.set_state(x=90, y=-80)