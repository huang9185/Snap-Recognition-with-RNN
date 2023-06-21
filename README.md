## Experiments

> Record 1

```
num_samples = 16000
target_sample_rate = 16000
learning_rate = 0.005
drop = 0.5

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=target_sample_rate,
    n_fft=400,
    hop_length=512,
    n_mels=84
)

sequence_length = 84
input_size = 32
output_size = 2
hidden_size = 2
num_layers = 5

optimizer = optim.Adam(model_0.parameters(),lr = learning_rate)
loss_fn = nn.BCELoss()
```

Below shows a graph plotting the losses over 2280 times of trainning for 10 epochs.

![A Graph](Assets/igures/Figure_1.png)

> Record 2

```
Params are not modified.
Model trained upon last record.
```

Below shows a graph plotting the losses over 2280 times of trainning for additional 10 epochs.

## Dateset
> this is a class
* annotation_file : the path of annotation file.
* audio_dir: the path of audio folder.
* transformation: function of mel_spectrogram.
* target_sample_rate: sample_rate of mel_spectrogram.
* num_samples: length of input audio per second.
#### functions
>__init__ <br/>
initialize different variables needed for the class
```code
def __init__(self,
                 annotation_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 ):  
        self.annotation_file = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples 
```
>__len__ <br/>
get number of datas in annotation file
```code
def __len__(self):
        return len(self.annotation_file)
```
>_get_sample_file_path <br/>
get file path of one audio file from annotation file.
```code
def _get_sample_file_path(self, index):
        path = os.path.join(self.audio_dir, str(self.annotation_file.iloc[index, 0]) + ".wav")
        return path
```
>get_sample_file_label <br/>
get label from annotation file
```code
def _get_sample_file_label(self, index):
        return self.annotation_file.iloc[index, 1]
```
>_resample <br/>
if sample rate is not target sample rate, resample the audio file.
```code
def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
```
>_mix_down <br/>
if there's more than one channel, get the mean of channels
```code
def _mix_down(self, signal):
        if (signal.shape[0] > 1):
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
```
>_cut_ <br/>
cut the length of wave tensor if it's longer than needed.
```code
def _cut_(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
```
>_right_pad <br/>
Fill wave tensor with zero entries from the right if length is shorter than needed
```code
def _right_pad(self, signal):
        if signal.shape[1] < self.num_samples:
            missing_samples = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, missing_samples))
        return signal
```
>__getitem__ <br/>
process data from dataloader.
```code
def __getitem__(self, index):
        signal, sr = torchaudio.load(self._get_sample_file_path(index))
        label = self._get_sample_file_label(index)
        # signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut_(signal)
        signal = self._right_pad(signal)
        signal= self.transformation(signal)
        return signal, label
```
![A Graph](Assets/Figures/Figure_2.png)

