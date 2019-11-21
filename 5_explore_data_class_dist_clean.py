import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

df = pd.read_csv('office_sounds.csv')
#print(df)
df.set_index('slice_file_name', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
print(df)

classes = list(np.unique(df.class_name))
class_dist = df.groupby(['class_name'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Dist', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()

df.reset_index(inplace=True)

