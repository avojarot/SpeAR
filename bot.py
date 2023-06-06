import os

import gantry
import openai
import pinecone
import telebot
from google.cloud import speech
from telebot import types

from src import Diarizer, MyPredictor, convert_to_wav
from src.data import create_index, upload_blob
from src.model.asr import AsrModel
from src.model.models.dolg import *

gantry_key = "zCrof0OjxV7UDkciA78iWGwuv3M"
pinecone_key = "9f2bd33c-77a8-4f13-a5fe-a168806ea881"
token = "6121471539:AAEPfxQU0ed14z0CQxgF57MLCRgkAd5rjSg"
openai_key = "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"

bot = telebot.TeleBot(token)
bot.delete_webhook()
pinecone.init(api_key=pinecone_key, environment="us-west1-gcp-free")
index = create_index()
gantry.init(gantry_key)
application = gantry.get_application("SpeAR_3")
openai.api_key = openai_key

predictor = MyPredictor()
posible_options = ["SpeAR Ua", "OpenAI", "Google Speech-to-text"]
concurrent_model = posible_options[0]


def convert_speech_to_text_openai(audio_filepath):
    with open(audio_filepath, "rb") as audio:
        is_not_done = True
        while is_not_done:
            try:
                transcript = openai.Audio.transcribe("whisper-1", audio)
                is_not_done = False
            except openai.error.RateLimitError:
                return "OpenAI error, try again later"
        return transcript["text"]


def convert_speech_to_text_google(audio_filepath, sr):
    client = speech.SpeechClient.from_service_account_file(
        "./credentials/spear-bot-388313-6a23d6901400.json"
    )
    with open(audio_filepath, "rb") as f:
        mp3_data = f.read()
    audio_file = speech.RecognitionAudio(content=mp3_data)
    config = speech.RecognitionConfig(sample_rate_hertz=sr, language_code="uk-UA")
    response = client.recognize(config=config, audio=audio_file)
    if len(response.results) < 1:
        return "GCP error"

    text = response.results[0].alternatives[0].transcript
    return text


def convert_speech_to_text_my(audio_filepath, user):
    text = predictor.predict(audio_filepath, user, index)
    return text


def speach2text(wav_file, sr, message):
    if concurrent_model == posible_options[0]:
        text = convert_speech_to_text_my(wav_file, message.from_user.id)

    elif concurrent_model == posible_options[1]:
        text = convert_speech_to_text_openai(wav_file)

    else:
        text = convert_speech_to_text_google(wav_file, sr)

    application.log(
        inputs=[{"record": wav_file, "model": concurrent_model}],
        outputs=[{"generation": text}],
    )
    upload_blob(wav_file)
    os.remove(wav_file)
    return text


@bot.message_handler(commands="start")
def initail_mock_up(message: types.Message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("OpenAI")
    btn2 = types.KeyboardButton("Google Speech-to-text")
    btn3 = types.KeyboardButton("SpeAR Ua")

    markup.add(btn1, btn2, btn3)
    bot.send_message(
        message.chat.id,
        f"Оберіть модель для розпізнавання мовлення. Поточна модель: {concurrent_model}",
        reply_markup=markup,
    )


@bot.message_handler(content_types=["text"])
def change_model(message: types.Message):
    text = message.text
    global concurrent_model
    if text in posible_options:
        concurrent_model = text


@bot.message_handler(content_types=["voice", "audio"])
def asr_in_voice(message: types.Message):
    voice = message.voice if message.voice else message.audio
    voice_file = bot.get_file(voice.file_id)
    files = convert_to_wav(bot, voice_file)
    text = ""
    for wav_file, sr in files:
        text += speach2text(wav_file, sr, message)
    bot.send_message(message.chat.id, text)


bot.polling(none_stop=True)
