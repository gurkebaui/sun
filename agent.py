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


    def _construct_internal_monologue_prompt(self, main_situation: str, context: str, state: Dict[str, float]) -> str:
        """Baut den Prompt für die interne Analyse mit der neuen Fokus-Struktur."""
        full_context = (
            f"Context:\n- My internal state is: Arousal={state['x']:.0f}, Valence={state['y']:.0f}.\n"
            f"- Additional context is: '{context}'\n"
            # Erinnerungen werden hier nicht mehr benötigt, da sie im nächsten Schritt relevanter sind.
        )
        task = f"\nTask: Based on the context, what is my key conclusion or plan regarding the main situation?\nMain Situation: '{main_situation}'\nInternal Thought:"
        return f"{full_context}\n{task}"

    def _construct_final_response_prompt(self, main_situation: str, internal_thought: str,
                                         memories: List[Dict[str, Any]]) -> str:
        """Baut den finalen Prompt mit Fokus auf die Haupt-Situation."""
        persona = "You are CAPA. You are helpful, factual, and concise. You must always respond in English."

        memory_section = "Relevant memories:\n"
        if memories:
            unique_memories = list({mem['text']: mem for mem in memories}.values())
            for mem in unique_memories:
                memory_section += f"- You remember that '{mem['text']}'\n"
        else:
            memory_section += "- None\n"

        context = (
            f"Your internal thought is: '{internal_thought}'.\n"
            f"{memory_section}"
        )
        task = f"\nTask: Based on your personality, thought, and memories, what is your immediate, direct response to the user's statement?\nUser's Statement: '{main_situation}'\nYour Response:"
        return f"{persona}\n\n{context}\n\n{task}"

    def run_inference_cycle(self, sensory_data: dict) -> (str, str):
        """
        FINALE VERSION: Implementiert die Fokus-Logik.
        """
        print("\n--- Beginn des Zwei-Stufen-Denkprozesses ---")
        current_state = self.asc.get_state()
        temp = modulate_temperature(current_state['x'], current_state['y'])

        # SCHRITT 1: FOKUS FINDEN (Priorisierung der Wahrnehmung)
        vision_text = sensory_data.get("vision", "")
        sound_text = sensory_data.get("sound", "")
        speech_text = sensory_data.get("speech", "")

        main_situation = ""
        other_context = []
        if speech_text:
            main_situation = speech_text
            other_context.append(vision_text)
            other_context.append(sound_text)
        elif vision_text:
            main_situation = vision_text
            other_context.append(sound_text)
        else:
            main_situation = sound_text

        context_str = ". ".join(filter(None, other_context))

        # Schritt 2: Relevante Erinnerungen basierend auf dem FOKUS abrufen
        print(f"[2] Rufe relevante Erinnerungen für den Fokus ab: '{main_situation}'...")
        relevant_memories = self.memory.query_relevant_memories(main_situation, n_results=3)

        # Schritt 3a: Kognitiver Schritt 1 - INTERNER MONOLOG
        print("[3a] Formuliere internen Gedanken...")
        internal_prompt = self._construct_internal_monologue_prompt(main_situation, context_str, current_state)
        internal_thought = self.pag.infer(prompt=internal_prompt, temperature=temp)

        # Schritt 3b: Kognitiver Schritt 2 - EXTERNE ANTWORT
        print("[3b] Formuliere externe Antwort basierend auf dem Gedanken...")
        final_prompt = self._construct_final_response_prompt(main_situation, internal_thought, relevant_memories)
        final_answer = self.pag.infer(prompt=final_prompt, temperature=temp)

        # Schritt 4 & 5: Aktion & Lernen
        print("[5] Kognitiver Prozess abgeschlossen. Erlebnis wird im Puffer gespeichert.")
        experience_summary = f"In the situation '{main_situation}', I thought '{internal_thought}' and responded: '{final_answer}'"
        self.experience_buffer.append({'text': experience_summary, 'metadata': current_state})

        return final_answer, final_prompt

    def handle_stimulus(self, stimulus: dict):
        if self.swhor.is_sleeping:
            is_danger = self.vigilance.filter_stimulus(stimulus)
            if is_danger:
                self.swhor.interrupt_sleep()
                self.asc.set_state(x=90, y=-80)