import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile
import librosa

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

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

if len(os.listdir('clean')) == 0:
    for f in tqdm(df.slice_file_name):
        signal, rate = librosa.load('audio/'+f, sr=16000)
        mask = envelope(signal, rate, 0.00001)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])