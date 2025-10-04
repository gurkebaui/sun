# agent.py
import torch
from pag.model import PAG_Model
from asc.core import AffectiveStateCore
from swhor.regulator import SWHoR
from vigilance.subsystem import VigilanceSubsystem
from memory.subsystem import MemorySubsystem
from modulation.functions import modulate_temperature
from typing import List, Dict, Any


class CAPA_Agent:
    def __init__(self, pag_model: PAG_Model, asc: AffectiveStateCore):
        self.pag = pag_model
        self.asc = asc
        self.swhor = SWHoR()
        self.vigilance = VigilanceSubsystem()
        self.memory = MemorySubsystem()

    def update(self):
        current_state = self.asc.get_state()
        wants_to_sleep = self.swhor.sleep_pressure > self.swhor.PRESSURE_THRESHOLD
        if wants_to_sleep and not self.swhor.is_sleeping:
            is_safe = self.vigilance.is_safe_to_sleep(current_state['x'])
            if is_safe:
                self.asc.set_state(x=-50, y=current_state['y'])
            else:
                pass  # Schlaf wird verhindert
        swhor_deltas = self.swhor.update(self.asc.get_state()['x'])
        self.asc.update_state(delta_x=swhor_deltas['delta_x'], delta_y=swhor_deltas['delta_y'])

    def handle_stimulus(self, stimulus: dict):
        if self.swhor.is_sleeping:
            is_danger = self.vigilance.filter_stimulus(stimulus)
            if is_danger:
                self.swhor.interrupt_sleep()
                self.asc.set_state(x=90, y=-80)

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
        # FINALE KORREKTUR: Definiere die Persönlichkeit des Agenten
        persona = "You are CAPA. You are helpful, factual, and concise. You do not ask questions back to the user. Your goal is to give the most direct and useful answer possible."

        context = (
            f"Your internal thought is: '{internal_thought}'.\n"
            f"The user's request is: '{situation_text}'."
        )
        task = "\nTask: Based on your personality and your internal thought, what is your immediate, direct response to the user?\nYour Response:"
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

        print("[4&5] Kognitiver Prozess abgeschlossen.")
        self.memory.add_experience(
            f"In the situation '{situation_text}', I thought '{internal_thought}' and responded: '{final_answer}'",
            metadata=current_state
        )

        return final_answer, final_prompt