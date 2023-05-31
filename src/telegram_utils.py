import os
import uuid

import librosa
import soundfile as sf

AUDIOS_DIR = "./audio/"


def convert_to_mp3(bot, file):
    file_format = file.file_path.split(".")[-1]
    mp3_filepath = f"{str(uuid.uuid4())}.mp3"
    ogg_filepath = f"{str(uuid.uuid4())}.{file_format}"

    downloaded_file = bot.download_file(file.file_path)

    with open(ogg_filepath, "wb") as new_file:
        new_file.write(downloaded_file)

    audio, sr = librosa.load(ogg_filepath)
    sf.write(mp3_filepath, audio, sr)
    os.remove(ogg_filepath)
    return mp3_filepath
