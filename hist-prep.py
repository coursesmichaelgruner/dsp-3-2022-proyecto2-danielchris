import pickle
import librosa
import librosa.display
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

width = 25
hop = 10
mels = 40
wav_fs = 16000 

audio_val_path = 'audios-validation/'
audio_trn_path = 'audios-training/'
audio_tst_path = 'audios-testing/'
val_spec_path = 'spectrograms-validation-images/'
trn_spec_path = 'spectrograms-training-images/'
tst_spec_path = 'spectrograms-testing-images/'


for obj_path in [trn_spec_path]:#val_spec_path,trn_spec_path,tst_spec_path]:

    if obj_path == val_spec_path:
        path = audio_val_path
    if obj_path == tst_spec_path:
        path = audio_tst_path
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
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_dB, x_axis='off', y_axis='off', sr=fs, fmax=wav_fs/2, ax=ax,cmap='gray');
            if len(audio) != fs:
                print(path+directory+'/'+f, len(audio))
            with open(obj_path+directory+'/'+f.replace(".wav",".png"),'wb') as fname:
                print(fname)
                plt.savefig(fname)
                plt.close()
                #plt.show()
                del S
                del S_dB
                del audio
