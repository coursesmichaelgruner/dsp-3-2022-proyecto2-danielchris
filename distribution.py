import os
import matplotlib.pyplot as plt

test_path  = 'audios-testing/'
val_path   = 'audios-validation/'
train_path = 'audios-training/'

for path in [test_path,val_path,train_path]:
    
    files_per_class = []
    audio_class = []
    
    dirs = os.listdir(path)

    for audioclass in dirs:
        files = os.listdir(path+audioclass)
        audio_class.append(audioclass)
        files_per_class.append(len(files))

    
    fig,ax = plt.subplots(figsize = (15,5))
    ax.set_title((path.split('/')[0]).replace('-',' for ') + " distribution")
    plt.bar(audio_class,files_per_class) 
    plt.savefig(path.replace('/','.png'))

