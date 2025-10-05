# pag/model.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings


class PAG_Model:
    """
    Predictive Action Generator (PAG) - Kognitives Upgrade.
    Nutzt ein vortrainiertes Flan-T5-Modell von Hugging Face für
    fortschrittliche Sprachverarbeitung.
    """

    def __init__(self, model_name: str = "google/flan-t5-xl"):
        """
        Initialisiert den PAG durch Laden des vortrainierten Modells und Tokenizers.
        Verschiebt das Modell automatisch auf die verfügbare GPU, falls vorhanden.

        Args:
            model_name (str): Der Name des Hugging Face Modells, das geladen werden soll.
        """
        print(f"Initialisiere kognitiven Kern...")
        print(f"Lade Modell: {model_name}...")

        # Gerät für die Ausführung bestimmen (GPU, falls verfügbar)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lade den Tokenizer und das Modell
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        print(f"PAG erfolgreich auf Gerät '{self.device}' initialisiert.")

    def infer(self, prompt: str, temperature: float) -> str:
        """
        Führt eine Inferenz mit dem geladenen Sprachmodell aus.
        Die Inferenz wird durch den Temperatur-Parameter aus dem ASC moduliert.

        Args:
            prompt (str): Die textbasierte Eingabeaufforderung für das Modell.
            temperature (float): Der Sampling-Temperaturwert. Höhere Werte
                                 führen zu kreativeren, niedrigere zu
                                 konservativeren Ausgaben. Muss > 0 sein.

        Returns:
            str: Die vom Modell generierte Textantwort.
        """
        # Sicherheitsprüfung: Die Temperatur für model.generate darf nicht 0 oder kleiner sein.
        if temperature <= 0:
            warnings.warn(f"Ungültige Temperatur {temperature} empfangen. Setze auf sicheren Mindestwert 0.1.")
            temperature = 0.1

        # 1. Tokenisierung: Wandle den Text-Prompt in numerische IDs um.
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # 2. Generierung: Erzeuge die Antwort-IDs.
        #    do_sample=True ist KRITISCH, um das Temperatur-Sampling zu aktivieren.
        #    Ohne diesen Parameter wird die Temperatur ignoriert.
        outputs = self.model.generate(
            input_ids,
            max_length=128,
            temperature=temperature,
            do_sample=True
        )

        # 3. Dekodierung: Wandle die Antwort-IDs zurück in lesbaren Text.
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_output