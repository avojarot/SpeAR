import os

import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from src import convert_to_mp3, download_voice

token = "6121471539:AAEPfxQU0ed14z0CQxgF57MLCRgkAd5rjSg"
openai_key = "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"
openai.api_key = openai_key

bot = Bot(token)
dispecher = Dispatcher(bot)


def convert_speech_to_text(audio_filepath):
    with open(audio_filepath, "rb") as audio:
        is_not_done = True
        while is_not_done:
            try:
                transcript = openai.Audio.transcribe("whisper-1", audio)
                is_not_done = False
            except Exception:
                is_not_done = True

        return transcript["text"]


@dispecher.message_handler(commands="start")
async def hello(message: types.Message):
    await message.answer("Hello")


@dispecher.message_handler(
    content_types=types.ContentTypes.VOICE | types.ContentTypes.AUDIO
)
async def asr_in_voice(message: types.Message):
    voice = message.voice if message.voice else message.audio
    voice_file = await voice.get_file()
    file_format = voice_file.file_path.split(".")[-1]

    ogg_file = await download_voice(voice_file, bot, file_format)
    mp3_file = convert_to_mp3(ogg_file, file_format)

    text = convert_speech_to_text(mp3_file)
    os.remove(ogg_file)
    os.remove(mp3_file)
    await message.answer(text)


executor.start_polling(dispecher, skip_updates=True)
