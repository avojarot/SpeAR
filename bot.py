import os

import docx
import gantry
import openai
import pinecone
import telebot
from google.cloud import speech
from telebot import types

from src import (
    Diarizer,
    MyPredictor,
    convert_to_wav,
    download_file_from_google_drive,
    download_from_telegram,
    time_now,
    video2audio,
)
from src.data import connect_with_connector, create_index, upload_blob
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
posible_models = ["SpeAR Ua", "OpenAI", "Google Speech-to-text"]
posible_types = [
    "Повідомлення",
    ".txt",
    ".docx",
    ".srt",
]
concurrent_model = posible_models[0]
export_type = posible_types[0]

engine = connect_with_connector()


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
    if concurrent_model == posible_models[0]:
        text = convert_speech_to_text_my(wav_file, message.from_user.id)

    elif concurrent_model == posible_models[1]:
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


def do_asr(message, voice_file):
    files = convert_to_wav(voice_file)
    text = ""
    for wav_file, sr in files:
        text += speach2text(wav_file, sr, message)
    if export_type == "Повідомлення":
        bot.send_message(message.chat.id, text)
    else:
        file_name = "asr_result" + str(time_now()) + export_type
        if export_type == ".docx":
            document = docx.Document()
            document.add_paragraph(text)
            document.save(file_name)
        else:
            with open(file_name, "w") as f:
                f.write(text)

        with open(file_name, "rb") as f:
            bot.send_document(message.chat.id, f)

        os.remove(file_name)


@bot.message_handler(content_types=["text"])
def change_model(message: types.Message):
    text = message.text
    global concurrent_model, export_type

    if text in posible_models:
        concurrent_model = text
        bot.send_message(
            message.chat.id,
            f"Змінено модель для розпізнавання мовлення на {concurrent_model}",
            reply_markup=get_markup(),
        )
    elif text in posible_types:
        export_type = text
        bot.send_message(
            message.chat.id,
            f"Змінено формат експорту на {export_type}",
            reply_markup=get_markup(),
        )
    elif text == "Вибір моделі":
        buttons = posible_models + ["Назад 🔙"]
        bot.send_message(
            message.chat.id,
            f"Оберіть модель для розпізнавання мовлення",
            reply_markup=get_markup(buttons),
        )
    elif text == "Вибір формату експорту":
        buttons = posible_types + ["Назад 🔙"]
        bot.send_message(
            message.chat.id,
            f"Оберіть формат експорту",
            reply_markup=get_markup(buttons),
        )

    elif text == "Назад 🔙":
        initail_mock_up(message)

    elif text.startswith("https://drive.google.com/file/d/"):
        text_len = len("https://drive.google.com/file/d/")
        file_id = text[text_len:].split("/")[0]
        file = download_file_from_google_drive(file_id)
        if file.endswith(".mp4"):
            voice_file = video2audio(file)
        elif file.endswith(".mp3") or file.endswith(".wav"):
            voice_file = file
        else:
            return
        do_asr(message, voice_file)


@bot.message_handler(content_types=["voice", "audio", "video"])
def asr_in_voice(message: types.Message):
    if message.video:
        video_file = bot.get_file(message.video.file_id)
        file = download_from_telegram(bot, video_file)
        voice_file = video2audio(file)
    else:
        voice = message.voice if message.voice else message.audio
        voice_file = download_from_telegram(bot, bot.get_file(voice.file_id))
    do_asr(message, voice_file)


def get_markup(buttons=["Вибір моделі", "Вибір формату експорту"]):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(*[types.KeyboardButton(i) for i in buttons])
    return markup


@bot.message_handler(commands="start")
def initail_mock_up(message: types.Message):
    bot.send_message(
        message.chat.id,
        f"Поточна модель: {concurrent_model}, поточний формат експорту {export_type}",
        reply_markup=get_markup(),
    )


bot.polling(none_stop=True)
