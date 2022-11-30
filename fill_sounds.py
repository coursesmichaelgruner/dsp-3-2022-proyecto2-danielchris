from pydub import AudioSegment as ads
import os
import numpy as np

"extending audios to 1s"

path_t = 'audios-training/' 
path_v = 'audios-validation/'

dirs_val = os.listdir(path_t)

newaudio = [];
for path in [path_t,path_v]:
    for directory in dirs_val:
        files = os.listdir(path+directory)
        for f in files:
        
            audio = ads.from_wav(path+directory+'/'+f)
            diff = 1000 - len(audio)
            if diff > 0:
                empty_seg = ads(np.zeros(diff*16,'<u2').tobytes(),
                                frame_rate = 16000,
                                sample_width = 2,
                                channels = 1)
                newaudio = audio.append(empty_seg,crossfade=0)
                newaudio.export(path+directory+'/'+f,format ="wav")
       
