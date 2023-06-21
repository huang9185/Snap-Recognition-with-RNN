
## Global Variables
> INPUT_SIZE : size of the input audio files <br/>
> HIDDEN_SIZE :  size of the hidden neurons<br/>
> NUM_LAYERS : number of layers in rnn<br/>
> LR : learning rate<br/>
> TARGET_SAMPLE_RATE : sample rate in mel-spectrogram<br/>
> NUM_SAMPLES : length of input audio file<br/>
> N_FFT : size of Fast Fourier Transformation <br/>
> HOP_LENGTH : length of hop between Short-Time Fourier Transformation windows<br/>
> N_MELS : number of mel filter banks<br/>
> SEQUENCE_LENGTH : length of input in rnn<br/>
> OUTPUT_SIZE : size of the output neuron<br/>
> CHUNK : number of frames in the buffer<br/>
> FORMAT : format of audio input <br/>
> CHANNELS : number of the channels of the device<br/>
> RATE : number of samples collected per second<br/>
> BUFFER_SIZE : frames per buffer<br/>
> RECORD_SECONDS : number of seconds to record in each audio<br/>
> WAVE_OUTPUT_FILENAME : the file to store the output in wave format<br/>
> DROP : probability to drop each neuron<br/>
> mel_spectrogram : used as a transformation from an audio file into a tensor<br/>
> p : an instance of pyaudio<br/>
> stream : an audio stream opened using pyaudio<br/>
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

## myModule
> This is a class
* num_layers : number of layers of the recurrent network
* input_size : the size of the input audio files
* hidden_size : number of hidden neurons
* output_size : the size of the output layer
* drop : probability to drop each neuron
* rnn : the recurrent neuron network
* fc : a linear network to reshape
  
#### functions
>__init__ <br/>
initialize variables
```code
def __init__(self,sequence_length,input_size,hidden_size,num_layers,output_size):
        super(myModule,self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop = nn.Dropout(p=DROP)
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,nonlinearity='tanh',batch_first=True)
        self.fc = nn.Linear(sequence_length,1)
```
>forward <br/>
modify the input variables into output using rnn, drop, fc, and sigmoid activation fucntion
```code
def forward(self,x,hidden):
        out_pre,hidden = self.rnn(x,hidden)
        out_pre = self.drop(out_pre)
        out = self.fc(out_pre.T)
        out = nn.Sigmoid()(out)
        return out
```

## Experiments : Trainning loop

>Prediction Function
This function predict the label of sample from output tensor.
```
def pred(y_pred,y):
  if(y_pred[0]>y_pred[1]):
    return 1-y
  else:
    return y
```
>Tranning loop
This loop through every sample in the dataset, predict the output and plot the correctness diagram.
```code
for i in range(epochs):
    for x,y in dataloader:
        hidden = torch.zeros(model_0.num_layers,model_0.hidden_size) # every time go through a sequence , initialize the hidden
        y_pred = model_0(torch.squeeze(x,1).reshape(40,400),hidden)
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
```
![A Graph](Assets/igures/Figure_1.png)

![A Graph](Assets/Figures/Figure_2.png)
