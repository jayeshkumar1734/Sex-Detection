import pyaudio
import wave
import torch
import torch
import torch.nn as nn
import torch
import librosa
import numpy as np
import pandas as pd
class LeNet(nn.Module):
    def __init__(self): 
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv1d(196, 32, 3),         
            nn.ReLU(),
            #nn.MaxPool2d(2, stride=2),  
            nn.Conv1d(32, 48, 3),        
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),   
            nn.Conv1d(24, 120, 3),         
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  
        )
        self.fc_model = nn.Sequential(
            nn.Linear(180,64),        
            nn.ReLU(),
            nn.Linear(64,2),
            nn.ReLU()           
        )
        self.size=180
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps=1e-9

       
        #nn.init.xavier_normal_(self.cnn_model.weight)
        #nn.init.xavier_normal_(self.fc_model.weight)
        
    def forward(self, x):
        
        x = self.cnn_model(x)
        #print("before fcc ",x)
        x = x.view(x.size(0), -1)
        x = self.alpha * (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        x = self.fc_model(x)
        #print("after fcc ",x)
        return x
def init_weights(m):
  
  if type(m) == nn.Linear or type(m)== nn.Conv2d:
    nn.init.xavier_normal_(m.weight)
    #print(m.weight)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = LeNet()
net.apply(init_weights)
resume_weights = './checkpoint_1.pth.tar'
import os.path
# If exists a best model, load its weights!
if os.path.isfile(resume_weights):
    #print("=> loading checkpoint '{}' ...".format(resume_weights))
    if device:
        checkpoint = torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(resume_weights,
                                map_location=lambda storage,
                                loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    net.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)",checkpoint['epoch'],best_accuracy,start_epoch)

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("finished recording")
 
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

y, sr = librosa.load("./file.wav",sr=None)
speech, _ = librosa.effects.trim(y)

speech=speech[:100000]
S1=librosa.feature.mfcc(y=speech,sr=sr)
S1=torch.tensor(S1)
S1=S1.float()
S1=S1.reshape(1,S1.shape[1],S1.shape[0])
# S1=S1.to(device)
out=net(S1)
print(out)
_,a=torch.max(out,1)
if a == 0:
    print("Female voice")
else:
    print("Male voice")
