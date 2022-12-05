import os
import shutil
import time

print("SORTING TEST AUDIO SAMPLES")


classes = ['yes','no','up','down','left','right','on','off','stop','go']

unk_classes = []

total = 0

if os.path.exists('audios-testing/unknown'):
    shutil.rmtree('audios-testing/unknown')

custom_testing_dir = os.listdir('test-audio')

for f in custom_testing_dir:
    current_class = f.split('-')[0]
    if not os.path.exists('audios-testing/'+current_class):
        os.mkdir('audios-testing/'+current_class)
    shutil.copyfile('test-audio/'+f,'audios-testing/'+current_class+'/'+f)

os.mkdir('audios-testing/unknown')

with open('testing_list.txt') as testing_list:
    counter = 0
    for line in testing_list:
        line = line.split('\n')[0]
        current_class = line.split('/')[0]
        if not current_class in classes:
            shutil.copyfile('audios-original/' + line, 'audios-testing/unknown/'+(line.split('/')[1]))
            counter+=1
            if counter == 10:
                break
