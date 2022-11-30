import os
import shutil
import time

print("SORTING TRAINING AUDIO SAMPLES")

classes = ['yes','no','up','down','left','right','on','off','stop','go']

unk_classes = []

total = 0

if os.path.exists('audios-training/unknown'):
    shutil.rmtree('audios-training/unknown')

with open("validation_list.txt") as f:
    
    vals = f.read();
    val_list = vals.split("\n")[:-1] 

    dirs = os.listdir('audios-original/')

    for directory in dirs:
        files = os.listdir('audios-original/'+directory)
        for f in files:
            if (directory +'/'+f) not in val_list:
                if not os.path.exists('audios-training/'+directory):
                    os.mkdir('audios-training/'+directory)
                shutil.copyfile('audios-original/'+ directory +'/'+ f, 'audios-training/'+ directory +'/'+ f)
   
dirs = os.listdir('audios-training/')
os.mkdir('audios-training/unknown')

for directory in dirs:
    if directory not in classes:
        files = os.listdir('audios-training/'+ directory)
        num = 0
        for f in files:
            path = 'audios-training/'+ directory + '/' + f
            newpath ='audios-training/unknown/'+ f
            if not os.path.exists(newpath):
                shutil.move(path,newpath)
                num += 1
            if num >= 130:
                break
        shutil.rmtree('audios-training/'+ directory)
