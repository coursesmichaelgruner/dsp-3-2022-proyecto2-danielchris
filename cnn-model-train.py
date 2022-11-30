import pickle
import librosa
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

classes = {'yes'        :0,
           'no'         :1,
           'up'         :2,
           'down'       :3,
           'left'       :4,
           'right'      :5,
           'on'         :6,
           'off'        :7,
           'stop'       :8,
           'go'         :9,
           'unknown'    :10,
           'background' :11,
        }

trn_set_path = 'spectrograms-training/'

x_train = []
y_train = []

dirs = os.listdir(trn_set_path)

for audio_class in dirs:
    files = os.listdir(trn_set_path+audio_class)
    for f in files:
        
        y_train.append(classes[audio_class])
        
        file_path = trn_set_path+audio_class+'/'+f
        with open(trn_set_path+audio_class+'/'+f,'rb') as fspec:
            #print(file_path)
            spectro = pickle.load(fspec)
            data_item = np.empty(list(spectro.shape)+[1])
            data_item[:,:,0] = spectro
            x_train.append(data_item)




y_train = np.asarray(y_train)
#y_train = np.concatenate(y_train,y_train)
y_train = keras.utils.to_categorical(y_train,num_classes=12)

x_train = np.stack(x_train)
print(y_train.shape)
print(x_train.shape)

#xshape = (len(x_train),len(x_train[0]),len(x_train[0][0]),len(x_train[0][0][0]))
#print(xshape)
print(x_train.shape[1:])

model = Sequential()
model.add(InputLayer(input_shape=x_train.shape[1:]))                   
model.add(Conv2D(12,3,padding="same"))                      
model.add(BatchNormalization())                             
model.add(ReLU())                                           
model.add(MaxPool2D(3,padding = "same",strides = (2,2)))
model.add(Conv2D(24,3,padding="same"))
model.add(BatchNormalization())                             
model.add(ReLU())                                           
model.add(MaxPool2D(3,padding = "same",strides = (2,2)))
model.add(Conv2D(48,3,padding="same"))      
model.add(BatchNormalization())                             
model.add(ReLU())                                           
model.add(MaxPool2D(3,padding = "same",strides = (2,2)))
model.add(Conv2D(48,3,padding="same"))      
model.add(BatchNormalization())                             
model.add(ReLU())                                           
model.add(Conv2D(48,3,padding="same"))      
model.add(BatchNormalization())                             
model.add(ReLU())                                           
model.add(MaxPool2D((1,13),strides = (1,1)))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(12))
model.add(Softmax())

opt = keras.optimizers.Adam(learning_rate=0.0003)
model.compile(loss ='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.summary()

if not (os.path.exists('spec_model')):
    model.fit(x_train,y_train,epochs=25,batch_size=10)
    model.save('spec_model.h5')
