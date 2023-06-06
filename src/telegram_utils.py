import os
import uuid

import librosa
import soundfile as sf

AUDIOS_DIR = "./audio/"
MAX_DURATION = 60


def convert_to_wav(bot, file):
    file_format = file.file_path.split(".")[-1]

    ogg_filepath = f"{str(uuid.uuid4())}.{file_format}"

    downloaded_file = bot.download_file(file.file_path)

    with open(ogg_filepath, "wb") as new_file:
        new_file.write(downloaded_file)

    audio, sr = librosa.load(ogg_filepath)
    files = []
    for i in range(0, len(audio), sr * MAX_DURATION):
        mp3_filepath = f"{str(uuid.uuid4())}.wav"
        sf.write(mp3_filepath, audio[i : i + sr * MAX_DURATION], sr)
        files.append((mp3_filepath, sr))
    os.remove(ogg_filepath)
    return files
