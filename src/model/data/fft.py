import torch
import torchaudio

from .constants import DURATION, SR

featurizer = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=1024,
    win_length=1024,
    hop_length=215,
    n_mels=224,
    window_fn=torch.hann_window,
    center=True,
    pad_mode="reflect",
    power=2.0,
)


def pad_audio(audio):
    target_len = len(audio) + (len(audio) % DURATION)
    audio_pad = torch.zeros(target_len)
    audio_pad[: len(audio)] = torch.tensor(audio)
    return audio_pad


def to_mel(audio):
    return featurizer(pad_audio(audio)).squeeze().clamp(1e-5).log()
