import os
import pyaudio
import wave
import csv

TrueSample = 148
FalseSample = 652

# Routes: to be changed for raspberry pi
atf = r'C:\Users\Elyn\Documents\AI\RNN Sound Recog\Version 3 - Local\LiveAnnotation.csv'
ad = r'C:\Users\Elyn\Documents\AI\RNN Sound Recog\Version 3 - Local\LiveAudio'

LABEL = 1
# Constants for live prediction
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
BUFFER_SIZE = 1024
DROP = 0.5

def record_save(duration, start, end, atf, ad):
    p = pyaudio.PyAudio()  # Instantiate PyAudio and initialize PortAudio system resources
    stream = p.open(format=FORMAT,  # Open stream
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=BUFFER_SIZE)
    count = start;
    print("Start")
    while(count<end+1):
        output_folder = ad
        filename = os.path.join(output_folder, str(count)+'.wav')
        print("The filename is ", filename)
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        count += 1
    p.terminate()
    with open(atf, 'a', newline='') as annotation:
        writer = csv.writer(annotation)
        for i in range(start, count):
            row = [i, LABEL]
            writer.writerow(row)

record_save(0.5, 896, 920, atf, ad)