import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import pyaudio
import wave

## Always change the x axis for the graph plotted = epochs * # dataset

# Routes: to be changed for raspberry pi
atf = r'#'
ad = r'#'

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
drop = 0.5

# Used in MyDataset
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

class MyDataset(Dataset):  # read audio file and return label and waveforms

    def __init__(self,
                 annotation_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 ):  # add device here
        self.annotation_file = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        # self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples  #

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        signal, sr = torchaudio.load(self._get_sample_file_path(index), format="wav") # data is a np array
        label = self._get_sample_file_label(index)
        # signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut_(signal)
        signal = self._right_pad(signal)
        signal= self.transformation(signal)
        return signal, label

    def _cut_(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad(self, signal):
        if signal.shape[1] < self.num_samples:
            missing_samples = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, missing_samples))
        return signal

    def _get_sample_file_path(self, index):
        path = os.path.join(self.audio_dir, str(self.annotation_file.iloc[index, 0]) + ".wav")
        return path

    def _get_sample_file_label(self, index):
        return self.annotation_file.iloc[index, 1]

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down(self, signal):
        if (signal.shape[0] > 1):
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


class myModule(nn.Module):
    def __init__(
        self,
        sequence_length,
        input_size,
        hidden_size,
        num_layers,
        output_size):
        super(myModule,self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop = nn.Dropout(p=drop)
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,nonlinearity='tanh',batch_first=True)
        self.fc = nn.Linear(sequence_length,1)

    def forward(self,x,hidden):
        out_pre,hidden = self.rnn(x,hidden)
        out_pre = self.drop(out_pre)
        out = self.fc(out_pre.T)
        out = nn.Sigmoid()(out)
        return out


def pred(y_pred,y):
  if(y_pred[0]>y_pred[1]):
    return 1-y
  else:
    return y

model_0 = myModule(
    sequence_length=SEQUENCE_LENGTH,
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS)
epochs = 50
optimizer = optim.Adam(model_0.parameters(),lr=LR)
dataset = MyDataset(atf,ad,mel_spectrogram,TARGET_SAMPLE_RATE, NUM_SAMPLES)
dataloader = DataLoader(dataset,batch_size = 1,shuffle = True)
loss_fn = nn.BCELoss()
rate = 1
sum = 1
losses = []
test_input = []
test_output= []
for i in range(epochs):
    for x,y in dataloader:
        hidden = torch.zeros(model_0.num_layers,model_0.hidden_size) # every time go through a sequence , initialize the hidden
        y_pred = model_0(x.squeeze(),hidden)
        loss = loss_fn(y_pred,torch.tensor([[1.0],[0.0]]))
        correct = pred(y_pred,y)
        # Snapping is True
        if y==1:
          loss = loss_fn(y_pred,torch.tensor([[0.0],[1.0]]))
          correct = pred(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum+=1
        rate+=correct
        losses.append(rate/sum)
        test_output.append(correct)
    print("Epoch "+str(i)+" finished")

torch.save(model_0.state_dict(), "#")
print("Finished")