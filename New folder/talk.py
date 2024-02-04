import sounddevice as sd
import soundfile as sf
import numpy as np
import openai
import os
import requests
import re
from colorama import Fore, Style, init
import datetime
import base64
from pydub import AudioSegment
from pydub.playback import play
from openaiapikey2 import openaiapikey
from elabapikey import elabapikey
import matplotlib.pyplot as plt
import IPython.display as ipd
import sys
import pyaudio

import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import glob

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import torchaudio
import time



from scipy.io.wavfile import write

init()

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/ljs_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("pretrained_ljs.pth", net_g, None)





def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

api_key = openaiapikey
elapikey = elabapikey

conversation1 = []  
chatbot1 = open_file('chatbot1.txt')

def chatgpt(api_key, conversation, chatbot, user_input, temperature=0.9, frequency_penalty=0.2, presence_penalty=0):
    openai.api_key = api_key
    conversation.append({"role": "user","content": user_input})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input)
    chat_response = completion['choices'][0]['message']['content']
    conversation.append({"role": "assistant", "content": chat_response})
    return chat_response

# def text_to_speech(text, voice_id, api_key):
#     url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
#     headers = {
#         'Accept': 'audio/mpeg',
#         'xi-api-key': api_key,
#         'Content-Type': 'application/json'
#     }
#     data = {
#         'text': text,
#         'model_id': 'eleven_monolingual_v1',
#         'voice_settings': {
#             'stability': 0.6,
#             'similarity_boost': 0.85
#         }
#     }
#     response = requests.post(url, headers=headers, json=data)
#     if response.status_code == 200:
#         with open('output.mp3', 'wb') as f:
#             f.write(response.content)
#         audio = AudioSegment.from_mp3('output.mp3')
#         play(audio)
#     else:
#         print('Error:', response.text)

def text_to_speech(text):
    stn_tst = get_text(text,  hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    # torchaudio.save(file_path, torch.tensor(audio), hps.data.sampling_rate)

    # # Convert the saved audio to MP3 format
    # audio_segment = AudioSegment.from_wav(file_path)
    # audio_segment.export(file_path + ".mp3", format="mp3")

    # Play the audio
    sd.play(audio, hps.data.sampling_rate)
    sd.wait()
    time.sleep(15)

def print_colored(agent, text):
    agent_colors = {
        "Anna:": Fore.YELLOW,
        "Phong": Fore.GREEN,
    }
    color = agent_colors.get(agent, "")
    print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")

voice_id1 = 'zrHiDhphv9ZnVXBqCLjz'




pretrained_models = {
    "EfficientConformerCTCSmall": "1MU49nbRONkOOGzvXHFDNfvWsyFmrrBam",
    "EfficientConformerCTCMedium": "1h5hRG9T_nErslm5eGgVzqx7dWDcOcGDB",
    "EfficientConformerCTCLarge": "1U4iBTKQogX4btE-S4rqCeeFZpj3gcweA"
}

pretrained_model = "EfficientConformerCTCSmall"


from EfficientConformer_master.EfficientConformer.functions import create_model


config_file = "EfficientConformer_master/EfficientConformer/configs/" + pretrained_model + ".json"

# Load model Config
with open(config_file) as json_config:
  config = json.load(json_config)

# PyTorch Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Create and Load pretrained model
model = create_model(config).to(device)
model.summary()
model.eval()
model.load(os.path.join("callbacks", pretrained_model, "checkpoints_swa-equal-401-450.ckpt"))


def record_voice(filename, duration, sample_rate, channels=1, format=pyaudio.paInt16):
    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")

    frames = []

    for i in range(int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        array_data = np.frombuffer(data, dtype=np.int16)
        frames.append(array_data)

    print("Recording complete.",'\n')

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames)

    # Denoise the audio
    # audio_data = apply_noise_gate(audio_data)
    # denoised_audio = denoise_audio(audio_data)
    # audio_data = reduce_noise(y=audio_data, sr=sample_rate)


    sf.write(filename, audio_data, sample_rate)


def record_and_transcribe(duration=6, fs=16000):
    output_filename = "recorded_voice.wav"
    record_voice(output_filename,duration=duration,sample_rate=fs)
    audio, sr = torchaudio.load("recorded_voice.wav")

    transcription = model.gready_search_decoding(audio.to(device), x_len=torch.tensor([len(audio[0])], device=device))[0]
    print_colored("Phong:", f"{transcription}\n\n")

    
    # with open(output_filename, "rb") as file:
    #     openai.api_key = api_key
    #     result = openai.Audio.transcribe("whisper-1", file)
    # transcription = result['text']
    # print(transcription)
    return transcription


while True:
    user_message = record_and_transcribe()
    response = chatgpt(api_key, conversation1, chatbot1, user_message)
    print_colored("Anna:", f"{response}\n\n")
    user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()
    text_to_speech(user_message_without_generate_image)
