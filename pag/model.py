# pag/model.py
import torch
import torch.nn as nn
import math
import random
from typing import Callable, Optional
import warnings


# --- PositionalEncoding Klasse bleibt unverändert ---
class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PAG_Model(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Wir verwenden die Layer direkt, um Layer Dropping zu ermöglichen
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.sos_token, self.eos_token, self.pad_token = 0, 1, 2

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(sz)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                # --- Neue Parameter für die Hooks ---
                attention_mod_func: Optional[Callable] = None,
                layer_drop_rate: float = 0.0) -> torch.Tensor:

        if attention_mod_func is not None:
            # HOOK: Attention-Modulation
            # HINWEIS: Dies ist eine konzeptionelle Platzierung. Eine echte Implementierung
            # erfordert eine benutzerdefinierte MultiHeadAttention-Klasse.
            warnings.warn(
                "Attention modulation is not fully implemented in the standard nn.Transformer and serves as a conceptual hook.")

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        # --- Implementierung des Layer Dropping für den Encoder ---
        memory = src_emb
        for layer in self.transformer_encoder.layers:
            if self.training and random.random() < layer_drop_rate:
                continue  # Überspringe den Layer
            memory = layer(memory, src_key_padding_mask=src_padding_mask)

        # --- Implementierung des Layer Dropping für den Decoder ---
        output = tgt_emb
        for layer in self.transformer_decoder.layers:
            if self.training and random.random() < layer_drop_rate:
                continue  # Überspringe den Layer
            output = layer(output, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_padding_mask,
                           memory_key_padding_mask=src_padding_mask)

        return self.fc_out(output)

    def infer(self, input_sequence: torch.Tensor,
              max_len: int = 20,
              temperature: float = 1.0,
              attention_mod_func: Optional[Callable] = None,
              layer_drop_rate: float = 0.0):
        self.eval()
        device = next(self.parameters()).device
        src = input_sequence.unsqueeze(1).to(device)
        src_padding_mask = (src == self.pad_token).transpose(0, 1).to(device)

        # Encoder-Durchlauf (mit potenziellem Layer Dropping)
        memory = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.transformer_encoder.layers:
            if random.random() < layer_drop_rate:
                continue
            memory = layer(memory, src_key_padding_mask=src_padding_mask)

        ys = torch.ones(1, 1).fill_(self.sos_token).type_as(input_sequence.data).to(device)

        for i in range(max_len - 1):
            tgt_padding_mask = torch.zeros(ys.shape[1], ys.shape[0], device=device).bool()
            tgt_mask = self._generate_square_subsequent_mask(ys.size(0)).to(device)

            # Decoder-Durchlauf (ohne Layer Dropping in der Inferenz für Konsistenz)
            out = self.pos_encoder(self.embedding(ys) * math.sqrt(self.d_model))
            out = self.transformer_decoder(out, memory, tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_padding_mask,
                                           memory_key_padding_mask=src_padding_mask)
            out = self.fc_out(out)

            # HOOK: Temperatur-Sampling
            last_word_logits = out[-1, :, :] / temperature

            # Von argmax zu multinomial sampling wechseln, um Kreativität zu ermöglichen
            probs = torch.nn.functional.softmax(last_word_logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1).squeeze(1)

            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
            if next_word.item() == self.eos_token:
                break
        return ys.squeeze(1)