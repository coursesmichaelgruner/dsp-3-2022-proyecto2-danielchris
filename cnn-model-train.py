import pickle
import librosa
import numpy as np
import os
import psutil

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

def cpu_usage(y_true, y_pred):
    return psutil.cpu_percent()



trn_set_path = 'spectrograms-training/'
val_set_path = 'spectrograms-validation/'

x_train = []
y_train = []

x_val = []
y_val = []

dirs = os.listdir(trn_set_path)

for audio_class in dirs:
    files = os.listdir(trn_set_path+audio_class)
    for f in files:
        
        y_train.append(classes[audio_class])
        
        file_path = trn_set_path+audio_class+'/'+f
        with open(file_path,'rb') as fspec:
            #print(file_path)
            spectro = pickle.load(fspec)
            data_item = np.empty(list(spectro.shape)+[1])
            data_item[:,:,0] = spectro
            x_train.append(data_item)

for audio_class in dirs:
    files = os.listdir(val_set_path+audio_class)
    for f in files:
        
        y_val.append(classes[audio_class])
        
        file_path = val_set_path+audio_class+'/'+f
        with open(file_path,'rb') as fspec:
            spectro = pickle.load(fspec)
            data_item = np.empty(list(spectro.shape)+[1])
            data_item[:,:,0] = spectro
            x_val.append(data_item)


y_train = np.asarray(y_train)
y_val =np.asarray(y_val)


print(y_val.shape)

y_train = keras.utils.to_categorical(y_train,num_classes=12)
y_val = keras.utils.to_categorical(y_val,num_classes=12)

x_train = np.stack(x_train)
x_val = np.stack(x_val)

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
model.compile(loss ='categorical_crossentropy',optimizer=opt,metrics=['accuracy',cpu_usage])

model.summary()
keras.utils.plot_model(model,"model-cnn-spectrogram.png",show_shapes=True)

if not (os.path.exists('spec_model.h5')):
    history = model.fit(x_train,y_train,epochs=25,batch_size=10)
    model.save('spec_model.h5')
    _,loss, accuracy = model.evaluate(x_val,y_val)

else:
    loaded_model = keras.models.load_model("spec_model.h5", custom_objects = {'cpu_usage': cpu_usage})
    _,loss, accuracy = loaded_model.evaluate(x_val,y_val)

