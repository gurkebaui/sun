# perception/subsystem.py
import cv2
import sounddevice as sd
import numpy as np
import whisper
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import torch
import warnings

# Unterdrücke laute Warnungen
warnings.filterwarnings("ignore", category=UserWarning)


class PerceptionSubsystem:
    def __init__(self):
        print("Initializing Sensory Organs (Perception Subsystem)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vision_enabled = True
        self.audio_enabled = True

        try:
            print("Loading visual model (BLIP)...")
            self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base").to(self.device)
            cam = cv2.VideoCapture(0)
            if not cam.isOpened(): raise ConnectionError("Webcam not found.")
            cam.release()
            print("Visual perception initialized.")
        except Exception as e:
            print(f"ERROR during visual perception initialization: {e}")
            self.vision_enabled = False

        try:
            print("Loading auditory models (Whisper & Hugging Face Audio Classification)...")
            self.whisper_model = whisper.load_model("base", device=self.device)
            self.sound_classifier = pipeline(
                "audio-classification",
                model="superb/hubert-large-superb-er",
                device=0 if self.device.type == 'cuda' else -1
            )

            devices = sd.query_devices()
            if not any(d['max_input_channels'] > 0 for d in devices):
                raise ConnectionError("No active microphone found.")

            print("Auditory perception initialized.")
        except Exception as e:
            print(f"ERROR during auditory perception initialization: {e}")
            self.audio_enabled = False

    def _perceive_vision(self) -> str:
        if not self.vision_enabled: return "Visual perception is disabled."
        cam = cv2.VideoCapture(0)
        if not cam.isOpened(): return "Error: Could not access webcam."
        ret, frame = cam.read()
        cam.release()
        if not ret: return "Error: Could not capture image."
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.vision_processor(images=rgb_frame, return_tensors="pt").to(self.device)
        caption = self.vision_processor.decode(self.vision_model.generate(**inputs, max_length=50)[0],
                                               skip_special_tokens=True)
        return f"I see: {caption}."

    def _perceive_audio(self) -> (str, str):
        """
        FINALE VERSION: Radikal vereinfachte und stabile Audio-Aufnahme.
        """
        if not self.audio_enabled:
            return "Auditory perception is disabled.", ""
        try:
            samplerate = 16000
            duration = 5  # Ein fester, großzügiger 5-Sekunden-Aufnahmezeitraum.

            print(f"Listening for {duration} seconds...")
            audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            audio_data = audio_data.flatten()

            # Transkription
            transcription = self.whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available())['text'].strip()

            # Geräuscherkennung
            sound_results = self.sound_classifier(audio_data, top_k=1)
            inferred_class = sound_results[0]['label'] if sound_results else "Silence"

            sound_desc = f"I hear ambient sounds like: {inferred_class}."
            speech_desc = f"I hear someone say: '{transcription}'" if transcription else ""
            return sound_desc, speech_desc
        except Exception as e:
            print(f"WARNING: A non-critical error occurred during audio processing: {e}")
            return "Audio processing temporarily failed.", ""

    def perceive(self) -> dict:
        """
        FINALE VERSION: Gibt einen strukturierten Dictionary mit den getrennten
        Sinnesdaten zurück, anstatt eines formatierten Strings.
        """
        print("\n--- Perception Cycle ---")

        vision_report = self._perceive_vision()
        sound_report, speech_report = self._perceive_audio()

        # Gib die Rohdaten als strukturiertes Objekt zurück
        sensory_data = {
            "vision": vision_report,
            "sound": sound_report,
            "speech": speech_report
        }

        # Formatiere den Output nur für die Konsole
        print("Sensory Input:")
        for key, value in sensory_data.items():
            if value: print(f"- {key.capitalize()}: {value}")

        return sensory_data