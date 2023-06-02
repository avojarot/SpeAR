import numpy as np
import torch

from .constants import max_amp, min_amp
from .fft import to_mel


def preprosess(audio):
    mel = to_mel(audio)
    image = torch.Tensor(mel).numpy()
    image = np.clip((image - min_amp) / (max_amp - min_amp), 0, 1)
    return torch.from_numpy(image).unsqueeze(0)
