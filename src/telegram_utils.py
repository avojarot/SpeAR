import datetime
import os
import uuid

import librosa
import moviepy.editor as mp
import soundfile as sf

AUDIOS_DIR = "./audio/"
MAX_DURATION = 1200


def download_from_telegram(bot, file):
    file_format = file.file_path.split(".")[-1]

    ogg_filepath = f"{str(uuid.uuid4())}.{file_format}"

    downloaded_file = bot.download_file(file.file_path)

    with open(ogg_filepath, "wb") as new_file:
        new_file.write(downloaded_file)

    return ogg_filepath


def video2audio(filepath):
    clip = mp.VideoFileClip(filepath)
    audio_filepath = f"{str(uuid.uuid4())}.wav"

    clip.audio.write_audiofile(audio_filepath)
    clip.close()
    os.remove(filepath)
    return audio_filepath


def convert_to_wav(ogg_filepath):
    audio, sr = librosa.load(ogg_filepath)
    files = []
    for i in range(0, len(audio), sr * MAX_DURATION):
        mp3_filepath = f"{str(uuid.uuid4())}.wav"
        sf.write(mp3_filepath, audio[i : i + sr * MAX_DURATION], sr)
        files.append((mp3_filepath, sr))
    os.remove(ogg_filepath)
    return files


def time_now():
    return str(datetime.datetime.now().isoformat()).split(".")[0].replace(":", "-")
