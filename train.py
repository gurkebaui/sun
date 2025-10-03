# pag/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from pag.model import PAG_Model
import random

# --- Hyperparameter und Konfiguration ---
VOCAB_SIZE = 50  # Zahlen von 3 bis 49 + 3 spezielle Tokens
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 0.0001
EPOCHS = 2600
BATCH_SIZE = 32
NUM_SAMPLES = 1000

# Spezielle Tokens
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


def generate_data(n_samples, max_len=15):
    """Erzeugt Paare von Zahlenfolgen und deren Umkehrungen."""
    data = []
    for _ in range(n_samples):
        length = random.randint(5, max_len)
        seq = [random.randint(3, VOCAB_SIZE - 1) for _ in range(length)]
        src = [SOS_TOKEN] + seq + [EOS_TOKEN]
        tgt = [SOS_TOKEN] + seq[::-1] + [EOS_TOKEN]
        data.append((torch.tensor(src), torch.tensor(tgt)))
    return data


def create_batch(data_batch):
    """Verarbeitet einen Batch von Sequenzen, fügt Padding hinzu und erstellt Masken."""
    src_batch, tgt_batch = [], []
    for (src_item, tgt_item) in data_batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)

    src_batch = pad_sequence(src_batch, padding_value=PAD_TOKEN)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_TOKEN)

    return src_batch, tgt_batch


def main():
    """Hauptfunktion für das Training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    model = PAG_Model(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    train_data = generate_data(NUM_SAMPLES)

    print("Beginne Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        # Mische die Daten für jede Epoche
        random.shuffle(train_data)

        for i in range(0, len(train_data), BATCH_SIZE):
            batch_data = train_data[i:i + BATCH_SIZE]
            src, tgt = create_batch(batch_data)
            src, tgt = src.to(device), tgt.to(device)

            # Ziel für den Loss ist das Target-Tensor ohne SOS-Token
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            src_padding_mask = (src == PAD_TOKEN).transpose(0, 1)
            tgt_padding_mask = (tgt_input == PAD_TOKEN).transpose(0, 1)

            optimizer.zero_grad()

            output = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

            # Loss-Berechnung erfordert Umformung der Tensoren
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(train_data) / BATCH_SIZE)
        print(f"Epoche [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Speichere das trainierte Modell
    torch.save(model.state_dict(), "pag/pag_model.pth")
    print("Modell erfolgreich in pag/pag_model.pth gespeichert.")


if __name__ == '__main__':
    main()