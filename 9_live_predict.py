import pickle
import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score
import librosa
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "file.wav"
 

df = pd.read_csv('office_sounds.csv')
classes = list(np.unique(df.class_name))
fn2class = dict(zip(df.slice_file_name, df.class_name))
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

def get_sample(file_output):
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
    
    waveFile = wave.open('live_audio/'+file_output, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

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


def clean_data(file_name):
    signal, rate = librosa.load('live_audio/'+file_name, sr=16000)
    mask = envelope(signal, rate, 0.0003)
    wavfile.write(filename='live_clean/'+file_name, rate=rate, data=signal[mask])

def build_predictions(file_name):
    y_pred = []
    fn_prob = {}

    #print('Extracting features from audio')
#    for fn in tqdm(os.listdir(audio_dir)):
    rate, wav = wavfile.read('live_clean/'+file_name)
    y_prob = []
#        if wav.shape[0] > config.step:
    for i in range(0, wav.shape[0]-config.step, config.step):
        sample = wav[i:i+config.step]
        x = mfcc(sample, rate, numcep=config.nfeat, 
                    nfilt=config.nfilt, nfft=config.nfft).T
        x = (x - config.min) / (config.max - config.min)

        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))

    fn_prob[file_name] = np.mean(y_prob, axis=0).flatten()
    return y_pred, fn_prob

while(1):
    get_sample(WAVE_OUTPUT_FILENAME)
    clean_data(WAVE_OUTPUT_FILENAME)
    y_pred, fn_prob = build_predictions(WAVE_OUTPUT_FILENAME)
    #print(zip(classes, fn_prob))
    print(classes[np.argmax(fn_prob[WAVE_OUTPUT_FILENAME])])
    os.remove('live_clean/'+WAVE_OUTPUT_FILENAME)
    os.remove('live_audio/'+WAVE_OUTPUT_FILENAME)