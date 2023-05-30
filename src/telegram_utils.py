import os
import uuid

import librosa
import soundfile as sf

AUDIOS_DIR = "./audio/"


async def handle_file(bot, file, file_name: str, path: str):
    await bot.download_file(file_path=file.file_path, destination=f"{path}/{file_name}")


async def download_voice(voice, bot, format):
    ogg_filepath = f"{voice.file_id}.{format}"
    await handle_file(bot, voice, ogg_filepath, AUDIOS_DIR)
    return os.path.join(AUDIOS_DIR, ogg_filepath)


def convert_to_mp3(ogg_filepath, format):
    mp3_filepath = os.path.join(AUDIOS_DIR, f"{str(uuid.uuid4())}.mp3")
    audio, sr = librosa.load(ogg_filepath)
    sf.write(mp3_filepath, audio, sr)
    return mp3_filepath
