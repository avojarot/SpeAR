import os

import openai
import telebot
from telebot import types

from src import convert_to_mp3

token = "6121471539:AAEPfxQU0ed14z0CQxgF57MLCRgkAd5rjSg"
openai_key = "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"
openai.api_key = openai_key

bot = telebot.TeleBot(token)
bot.delete_webhook()


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


@bot.message_handler(commands="start")
def hello(message: types.Message):
    bot.send_message(message.chat.id, "Hello")


@bot.message_handler(content_types=["voice", "audio"])
def asr_in_voice(message: types.Message):
    voice = message.voice if message.voice else message.audio
    voice_file = bot.get_file(voice.file_id)
    mp3_file = convert_to_mp3(bot, voice_file)
    text = convert_speech_to_text(mp3_file)
    os.remove(mp3_file)
    bot.send_message(message.chat.id, text)


bot.polling(none_stop=True)
