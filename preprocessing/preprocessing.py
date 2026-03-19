import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def normalize_epochs_1(epochs):
    normalized = epochs.copy()
    for ch in range(epochs.shape[1]):
        mean = epochs[:, ch, :].mean()
        std = epochs[:, ch, :].std()
        if std > 0:
            normalized[:, ch, :] = (epochs[:, ch, :] - mean) / std
    return normalized


def concatenate_epochs_16(epochs, labels, n_classes=16, seq_length=16):
    """
    Склеиваем 16 эпох для многоклассовой классификации

    Для каждого символа берем 16 эпох (по одной на каждую букву)
    Возвращает: (n_sequences, seq_length, n_channels, n_timesteps)
    и target индекс (0-15)
    """
    n_epochs = len(epochs)
    n_sequences = n_epochs // seq_length

    sequences = []
    targets = []

    for i in range(n_sequences):
        seq = epochs[i*seq_length:(i+1)*seq_length]
        # target — та эпоха, где label=1
        target_idx = np.where(labels[i*seq_length:(i+1)*seq_length] == 1)[0]

        if len(target_idx) == 1:  # должна быть ровно одна target
            sequences.append(seq)
            targets.append(target_idx[0])

    return np.array(sequences), np.array(targets)


