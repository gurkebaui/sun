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
            self.memory = MemorySubsystem()  # Ruft die korrekte, persistente DB auf
            self.experience_buffer: List[Dict[str, Any]] = []
            self.perception = PerceptionSubsystem()
            self._previous_y = self.asc.get_state()['y']

    def _consolidate_and_synthesize_memories(self):
        print("\n=== BEGINN DER SCHLAFPHASE: KONSOLIDIERUNG & SYNTHESE ===")
        if not self.experience_buffer:
            print("Keine neuen Erlebnisse zum Konsolidieren. Schlaf ist rein erholsam.")
            print("===================== ENDE DER SCHLAFPHASE =====================")
            return

        print(f"Konsolidiere {len(self.experience_buffer)} Erlebnisse aus dem Kurzzeitgedächtnis...")
        current_state_on_sleep = self.asc.get_state()
        for experience in self.experience_buffer:
            self.memory.add_experience(experience['text'], experience['metadata'])

        print("Kurzzeitgedächtnis erfolgreich ins Langzeitgedächtnis verschoben.")
        self.experience_buffer.clear()

        print("Beginne Synthese-Phase (Träumen)...")
        recent_memories = self.memory.get_latest_memories(n_results=10)

        # KORREKTUR 1: f-string-Syntaxfehler behoben
        memory_texts = [f"- {mem['text']}" for mem in recent_memories]
        memory_list_as_string = "\n".join(memory_texts)

        synthesis_prompt = (
            "You are in a sleep state, processing recent memories to find patterns.\n\n"
            "## RECENT EXPERIENCES ##\n"
            f"{memory_list_as_string}\n\n"
            "## TASK ##\n"
            "Analyze these experiences. Summarize the single most important lesson or recurring pattern in one concise sentence. This summary will become a new core memory."
        )

        learned_lesson = self.pag.infer(prompt=synthesis_prompt, temperature=0.5)

        print(f"Synthese abgeschlossen. Gelernte Lektion: '{learned_lesson}'")
        self.memory.add_experience(
            learned_lesson,
            metadata={'type': 'synthesis', 'x': current_state_on_sleep['x'], 'y': current_state_on_sleep['y']}
        )
        print("Gelernte Lektion als neue Kern-Erinnerung gespeichert.")
        print("===================== ENDE DER SCHLAFPHASE =====================")

    def _update_arousal_from_valence_change(self):
            """
            FINALE KORREKTUR 2: Diese Methode wurde hinzugefügt.
            Der interne Reflex, der Arousal und Valenz koppelt.
            """
            current_y = self.asc.get_state()['y']
            y_delta = current_y - self._previous_y

            threshold = 5.0

            if y_delta < -threshold:
                self.asc.update_state(delta_x=10)
                print("[Internal Reflex] Negativer Valenz-Abfall erkannt. Arousal (Stress) erhöht.")
            elif y_delta > threshold:
                self.asc.update_state(delta_x=-10)
                print("[Internal Reflex] Positiver Valenz-Anstieg erkannt. Arousal (Stress) verringert.")

            self._previous_y = current_y

    def update(self, verbose: bool = False, text_override: str = None):
        """
        FINALE ARCHITEKTUR: Die zentrale "Bewusstseins"-Schleife.
        Trennt nun korrekt zwischen wachen und schlafenden Zuständen.
        """
        # 1. Autonome, unbewusste Prozesse (Herzschlag, Müdigkeit, Reflexe)
        self._run_background_processes()

        # 2. Prüfe den Bewusstseinszustand AM ANFANG des Zyklus
        if self.swhor.is_sleeping:
            print("Agent schläft... zZz...")
            return  # BEENDE den bewussten Zyklus hier.

        # --- WACHZUSTAND ---
        sensory_report = ""
        if text_override:
            print(f"Manuelle Eingabe empfangen: '{text_override}'")
            vision_report = self.perception._perceive_vision()
            sound_report = "I hear ambient sounds like: Speech."
            speech_report = f"I hear someone say: '{text_override}'"
            sensory_report = f"Sensory Input:\n- {vision_report}\n- {sound_report}\n- {speech_report}"
            print(sensory_report)
        else:
            sensory_report = self.perception.perceive()

        final_answer, full_prompt = self.run_inference_cycle(sensory_report)

        if verbose:
            print("\n" + "=" * 20 + " DEBUG: VOLLSTÄNDIGER PROMPT " + "=" * 20)
            print(full_prompt)
            print("=" * 66 + "\n")

        print(f"Agent > {final_answer}")

    def _run_background_processes(self):
        """FINALE KORREKTUR: Stellt die korrekte Reihenfolge der Schlaf-Logik sicher."""
        current_state = self.asc.get_state()
        self._update_arousal_from_valence_change()
        wants_to_sleep = self.swhor.sleep_pressure > self.swhor.PRESSURE_THRESHOLD

        # Führe zuerst das SWHoR-Update aus, um den Schlafzustand korrekt zu setzen
        swhor_deltas = self.swhor.update(self.asc.get_state()['x'])
        self.asc.update_state(delta_x=swhor_deltas['delta_x'], delta_y=swhor_deltas['delta_y'])

        # Prüfe JETZT, ob der Agent einschlafen will UND der Zustand sicher ist
        if wants_to_sleep and not self.swhor.is_sleeping:
            is_safe = self.vigilance.is_safe_to_sleep(current_state['x'])
            if is_safe:
                print("[Agent] Hoher Schlafdruck und sichere Umgebung. Leite Schlaf ein.")
                self.asc.set_state(x=-50, y=current_state['y'])
                # Informiere den SWHoR über den neuen Zustand, um `is_sleeping` zu synchronisieren
                self.swhor.update(self.asc.get_state()['x'])
                # Führe die Konsolidierung aus, NACHDEM der Schlafzustand gesetzt wurde
                self._consolidate_and_synthesize_memories()


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