import os.path

import librosa
import torch
from tqdm import tqdm

from .downloading import download_file_from_google_drive
from .model import Diarizer
from .model.asr import AsrModel
from .model.models.dolg import *


class MyPredictor:
    def __init__(self, diarizer_path="./best_model_18.pth", size=16_000):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(self.device)
        if not os.path.exists(diarizer_path):
            download_file_from_google_drive()

        self.diarizer = Diarizer(diarizer_path, 224, 224, 512, self.device, 0.1)

        self.asr = AsrModel(self.device)
        self.size = size

    def prepare_audio(self, audio, orig_sr, target_sr):
        audio = librosa.to_mono(audio)
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def predict(self, audio):

        audio, sr = librosa.load(audio)
        audio = self.prepare_audio(audio, sr, 16_000)
        sub, starts = self.diarizer.diarize(audio)
        sub = sub.split("\n\n")
        print(starts)
        for i in tqdm(range(len(starts) - 1)):
            curr = audio[starts[i] * self.size : starts[i + 1] * self.size]
            words = self.asr.transcribe(curr, None)
            sub[i] += "\n" + words
            print(sub[i])
        sub = "\n\n".join(sub).strip()
        return sub
