# perception/subsystem.py
import cv2
import sounddevice as sd
import numpy as np
import whisper
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import warnings
import pandas as pd


# ... (Warnungen unterdrücken) ...

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
            if not cam.isOpened(): raise ConnectionError("Webcam not found or could not be opened.")
            cam.release()
            print("Visual perception initialized.")
        except Exception as e:
            print(f"ERROR during visual perception initialization: {e}")
            self.vision_enabled = False

        try:
            print("Loading auditory models (Whisper & YAMNet)...")
            self.whisper_model = whisper.load_model("base")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

            yamnet_classes_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
            self.yamnet_class_names = pd.read_csv(yamnet_classes_url)['display_name'].tolist()

            devices = sd.query_devices()
            if not any(d['max_input_channels'] > 0 for d in devices):
                raise ConnectionError("No active microphone found.")

            print("Auditory perception initialized.")
        except Exception as e:
            print(f"ERROR during auditory perception initialization: {e}")
            self.audio_enabled = False

    def _perceive_vision(self) -> str:
        if not self.vision_enabled:
            return "Visual perception is disabled."

        cam = cv2.VideoCapture(0)
        if not cam.isOpened(): return "Error: Could not access webcam."

        ret, frame = cam.read()
        cam.release()

        if not ret: return "Error: Could not capture image from webcam."

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.vision_processor(images=rgb_frame, return_tensors="pt").to(self.device)
        generated_ids = self.vision_model.generate(pixel_values=inputs.pixel_values, max_length=50)
        caption = self.vision_processor.decode(generated_ids[0], skip_special_tokens=True)
        return f"I see: {caption}."

    def _perceive_audio(self) -> (str, str):
        if not self.audio_enabled:
            return "Auditory perception is disabled.", ""

        try:
            samplerate = 16000
            duration = 4
            audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            audio_data = audio_data.flatten()

            transcription = self.whisper_model.transcribe(audio_data, fp16=False)['text'].strip()

            scores, _, _ = self.yamnet_model(audio_data)
            top_class_index = tf.argmax(scores)

            # FINALE KORREKTUR 1: Konvertiere den Tensor explizit in einen Python-Integer.
            inferred_class = self.yamnet_class_names[int(top_class_index)]

            sound_desc = f"I hear ambient sounds like: {inferred_class}."
            speech_desc = f"I hear someone say: '{transcription}'" if transcription else ""

            return sound_desc, speech_desc
        except Exception as e:
            print(f"WARNING: A non-critical error occurred during audio processing: {e}")
            # FINALE KORREKTUR 2: Gib eine saubere, englische Fehlermeldung zurück.
            return "Audio processing temporarily failed.", ""

    def perceive(self) -> str:
        print("\n--- Perception Cycle ---")

        vision_report = self._perceive_vision()
        sound_report, speech_report = self._perceive_audio()

        full_report = f"Sensory Input:\n- {vision_report}\n- {sound_report}"
        if speech_report:
            full_report += f"\n- {speech_report}"

        print(full_report)
        return full_report