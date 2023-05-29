import torch
import os
import torchaudio
import torch.nn as nn
from torch import optim
import pyaudio
import wave

# Constants for model and spectrograms
INPUT_SIZE = 32
HIDDEN_SIZE = 2
NUM_LAYERS = 5
LR = 0.005
TARGET_SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
N_FFT = 400
HOP_LENGTH = 512
N_MELS = 84
SEQUENCE_LENGTH = 84
OUTPUT_SIZE = 2

# Constants for live prediction
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
BUFFER_SIZE = 1024
RECORD_SECONDS = 0.2
WAVE_OUTPUT_FILENAME = "output.wav"
DROP = 0.5

# Methods Fixed

class myModule(nn.Module):
    def __init__(self,sequence_length,input_size,hidden_size,num_layers,output_size):
        super(myModule,self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop = nn.Dropout(p=DROP)
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,nonlinearity='tanh',batch_first=True)
        self.fc = nn.Linear(sequence_length,1)

    def forward(self,x,hidden):
        out_pre,hidden = self.rnn(x,hidden)
        out_pre = self.drop(out_pre)
        out = self.fc(out_pre.T)
        out = nn.Sigmoid()(out)
        return out

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

def realTime(f):  # this function transforms wav to tensor (84 * 32) / (32 * 84)
  signal,sr = torchaudio.load(f)
  if sr != TARGET_SAMPLE_RATE: # resample
      resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
      signal = resampler(signal)
  if (signal.shape[0] > 1): # mix down
            signal = torch.mean(signal, dim=0, keepdim=True)
  if signal.shape[1] > NUM_SAMPLES: #cut
      signal = signal[:, :NUM_SAMPLES]
  if signal.shape[1] < NUM_SAMPLES: # right pad
            missing_samples = NUM_SAMPLES - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, missing_samples))
  signal = mel_spectrogram(signal)
  return signal

def predict(signal): # this function predict y ;y is an integer
  with torch.inference_mode():
    hidden = torch.zeros(NUM_LAYERS,HIDDEN_SIZE)
    y = MODEL_0(signal.squeeze(),hidden)
    if y[1]>y[0]:
       print(True)

# Retrieve the model trained
MODEL_0 = myModule(
    sequence_length=SEQUENCE_LENGTH,
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS)
MODEL_0.load_state_dict(torch.load("model_0.pt"))

# Creating the live recording stream
p = pyaudio.PyAudio()  # Instantiate PyAudio and initialize PortAudio system resources
stream = p.open(format=FORMAT,  # Open stream
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=BUFFER_SIZE)
count = 0;

while(count<60):
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    predict(realTime("output.wav"))
    os.remove("output.wav")
    count += 1

p.terminate()

