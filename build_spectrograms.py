import pickle
import librosa
import librosa.display
import os
import shutil
import numpy as np

width = 25
hop = 10
mels = 40
wav_fs = 16000 

audio_val_path = 'audios-validation/'
audio_trn_path = 'audios-training/'
val_path = 'spectrograms-validation/'
trn_path = 'spectrograms-training/'


for obj_path in [val_path,trn_path]:

    if obj_path == val_path:
        path = audio_val_path
    else:
        path = audio_trn_path
    
    dirs = os.listdir(path)

    for directory in dirs:

        files = os.listdir(path+directory)
    
        if os.path.exists(obj_path+directory):
            shutil.rmtree(obj_path+directory)
    
        os.mkdir(obj_path+directory)
    
        for f in files:
            audio,fs = librosa.load(path+directory+'/'+f, sr = wav_fs)
            #print(len(audio))
            if len(audio) < fs:
                audio = np.concatenate((audio,np.zeros((fs-len(audio)),dtype='int')))
            elif len(audio) > fs:
                audio = audio[:fs]

            S = librosa.feature.melspectrogram(y = audio,
                                               sr = fs,
                                               n_mels = mels,
                                               n_fft = width*fs//1000,
                                               hop_length = hop*fs//1000)

            S_dB = librosa.power_to_db(S,ref = np.max)
            if len(audio) != fs:
                print(path+directory+'/'+f, len(audio))
            with open(obj_path+directory+'/'+f.replace(".wav",".pkl"),'wb') as fname:
                pickle.dump(S_dB,fname,pickle.HIGHEST_PROTOCOL)
