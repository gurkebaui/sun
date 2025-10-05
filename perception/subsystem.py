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
    # ... (__init__ bleibt gleich, außer der Log-Nachrichten) ...
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

    # ... (_perceive_vision bleibt gleich) ...
    def _perceive_vision(self) -> str:
        # ... (Code bleibt gleich) ...
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

            # FINALE KORREKTUR: Wandle den Tensor in ein NumPy-Array um und extrahiere dann den Python-Skalar.
            top_class_index = tf.argmax(scores).numpy().item()
            inferred_class = self.yamnet_class_names[top_class_index]

            sound_desc = f"I hear ambient sounds like: {inferred_class}."
            speech_desc = f"I hear someone say: '{transcription}'" if transcription else ""

            return sound_desc, speech_desc
        except Exception as e:
            print(f"WARNING: A non-critical error occurred during audio processing: {e}")
            return "Audio processing temporarily failed.", ""

    # ... (perceive-Methode bleibt gleich) ...
    def perceive(self) -> str:
        # ... (Code bleibt gleich) ...
        return full_report