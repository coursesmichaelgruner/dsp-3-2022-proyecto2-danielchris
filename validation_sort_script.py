import os
import shutil
import time

print("SORTING VALIDATION AUDIO SAMPLES")


classes = ['yes','no','up','down','left','right','on','off','stop','go']

unk_classes = []

total = 0

if os.path.exists('audios-validation/unknown'):
    shutil.rmtree('audios-validation/unknown')

with open("validation_list.txt") as f:
    
    vals = f.read();
    val_list = vals.split("\n")[:-1]     
    for line in val_list:
        current_class = line.split("/")[0] 
        
        if not os.path.exists('audios-validation/'+ current_class):
            os.mkdir('audios-validation/'+current_class)
        shutil.copyfile('audios-original/'+ line, 'audios-validation/'+ line)
   
dirs = os.listdir('audios-validation/')
os.mkdir('audios-validation/unknown')

for directory in dirs:
    if directory not in classes:
        files = os.listdir('audios-validation/'+ directory)
        num = 0
        for f in files:
            path = 'audios-validation/'+ directory + '/' + f
            newpath ='audios-validation/unknown/'+ f
            if not os.path.exists(newpath):
                shutil.move(path,newpath)
                num += 1
            if num >= 13:
                break
        shutil.rmtree('audios-validation/'+ directory)

