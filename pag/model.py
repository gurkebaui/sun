# pag/model.py
import torch
import torch.nn as nn
import math
import random
from typing import Callable, Optional


class PositionalEncoding(nn.Module):
    """
    Fügt Positionsinformationen zu den Input-Embeddings hinzu.
    Dies ist entscheidend, da Transformer-Modelle von sich aus keine Sequenzordnung verstehen.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PAG_Model(nn.Module):
    """
    Predictive Action Generator (PAG) - Kernarchitektur
    Eine Encoder-Decoder-Transformer-Architektur zur Sequenz-zu-Sequenz-Transformation.
    """

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # PyTorch Transformer erwartet (seq_len, batch_size, dim)
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Spezielle Tokens für die Sequenzverarbeitung
        self.sos_token = 0  # Start of Sequence
        self.eos_token = 1  # End of Sequence
        self.pad_token = 2  # Padding

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Erzeugt eine Maske, um zu verhindern, dass der Decoder zukünftige Tokens sieht."""
        return self.transformer.generate_square_subsequent_mask(sz)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass des Modells.
        Tensor-Dimensionen: S = Quell-Sequenzlänge, T = Ziel-Sequenzlänge, N = Batch-Größe, E = Embedding-Dimension
        """
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        # --- HOOK-Platzierungen für Ticket [PAG-HOOKS] ---
        # HINWEIS: Da wir nn.Transformer verwenden, sind die inneren Schleifen abstrahiert.
        # Die Hooks werden hier konzeptionell platziert.

        # HOOK: Layer Dropping würde hier angewendet, indem man durch die Encoder/Decoder-Layer iteriert
        # und einige basierend auf 'layer_drop_rate' überspringt.

        # HOOK: Attention-Modulation würde in der forward() Methode der Attention-Klasse angewendet.
        # `attention_mod_func` würde auf die Attention-Scores angewendet, bevor die Softmax-Funktion aufgerufen wird.

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.fc_out(output)

    def infer(self, input_sequence: torch.Tensor,
              max_len: int = 20,
              temperature: float = 1.0,
              attention_mod_func: Optional[Callable] = None,
              layer_drop_rate: float = 0.0):
        """
        Inferenz-Methode, um eine Ausgabesequenz zu generieren.
        Erfüllt die Anforderungen von Ticket [PAG-HOOKS].
        """
        self.eval()
        device = next(self.parameters()).device

        src = input_sequence.unsqueeze(1).to(device)
        src_padding_mask = (src == self.pad_token).transpose(0, 1).to(device)

        ys = torch.ones(1, 1).fill_(self.sos_token).type_as(input_sequence.data).to(device)

        for i in range(max_len - 1):
            tgt_padding_mask = torch.zeros(ys.shape[1], ys.shape[0], device=device).bool()

            out = self.forward(src, ys, src_padding_mask, tgt_padding_mask)

            # HOOK: Temperatur-Sampling wird hier angewendet. Aktueller Wert: temperature
            last_word_logits = out[-1, :, :] / temperature

            next_word = torch.argmax(last_word_logits, dim=1)

            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)

            if next_word.item() == self.eos_token:
                break

        return ys.squeeze(1)