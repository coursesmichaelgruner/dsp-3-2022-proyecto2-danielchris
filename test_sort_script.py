import os
import shutil
import time

print("SORTING TEST AUDIO SAMPLES")


classes = ['yes','no','up','down','left','right','on','off','stop','go']

unk_classes = []

total = 0

if os.path.exists('audios-test/unknown'):
    shutil.rmtree('audios-test/unknown')

with open("test_list.txt") as f:
    
    vals = f.read();
    val_list = vals.split("\n")[:-1]     
    for line in val_list:
        current_class = line.split("/")[0] 
        
        if not os.path.exists('audios-test/'+ current_class):
            os.mkdir('audios-test/'+current_class)
        shutil.copyfile('audios-original/'+ line, 'audios-test/'+ line)
   
dirs = os.listdir('audios-test/')
os.mkdir('audios-test/unknown')

for directory in dirs:
    if directory not in classes:
        files = os.listdir('audios-test/'+ directory)
        num = 0
        for f in files:
            path = 'audios-test/'+ directory + '/' + f
            newpath ='audios-test/unknown/'+ f
            if not os.path.exists(newpath):
                shutil.move(path,newpath)
                num += 1
            if num >= 13:
                break
        shutil.rmtree('audios-test/'+ directory)

