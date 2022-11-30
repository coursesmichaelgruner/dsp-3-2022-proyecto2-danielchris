from pydub import AudioSegment as ads
import os

print("SLICING BACKGROUND AUDIO SAMPLES")

path = '_background_noise_/'
newpath_t = 'audios-training/background/'
newpath_v = 'audios-validation/background/'

if not os.path.exists(newpath_t):
    os.mkdir(newpath_t)

if not os.path.exists(newpath_v):
    os.mkdir(newpath_v)

nfile = 0

files = os.listdir(path)
for f in files:
    audio = ads.from_wav(path+f)
    audio_len=len(audio)//(1000)
    for i in range(0,audio_len):
        audio_slice = audio[i*1000:(i+1)*1000]
        if(i % 10 == 0):
            audio_slice.export(newpath_v+f.replace('.wav',str(i)+".wav"), format="wav")
        else:    
            audio_slice.export(newpath_t+f.replace('.wav',str(i)+".wav"), format="wav")
        
