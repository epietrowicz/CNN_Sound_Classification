import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(3):
        axes[x].set_title(list(signals.keys())[i])
        axes[x].plot(list(signals.values())[i])
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(3):
        data = list(fft.values())[i]
        Y, freq = data[0], data[1]
        axes[x].set_title(list(fft.keys())[i])
        axes[x].plot(freq, Y)
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(3):
        axes[x].set_title(list(fbank.keys())[i])
        axes[x].imshow(list(fbank.values())[i],
                cmap='hot', interpolation='nearest')
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=30)
    i = 0
    for x in range(3):
        axes[x].set_title(list(mfccs.keys())[i])
        axes[x].imshow(list(mfccs.values())[i],
                cmap='hot', interpolation='nearest')
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

df = pd.read_csv('office_sounds.csv')
#print(df)
df.set_index('slice_file_name', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('audio/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
print(df)

classes = list(np.unique(df.class_name))
class_dist = df.groupby(['class_name'])['length'].mean()

df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.class_name == c].iloc[0,0]
    signal, rate = librosa.load('audio/'+wav_file, sr=44100)
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel

#plot_signals(signals)
#plt.show()

#plot_fft(fft)
#plt.show()

#plot_fbank(fbank)
#plt.show()

plot_mfccs(mfccs)
plt.show()
