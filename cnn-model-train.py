import pickle
import librosa
import numpy as np
import os
import psutil

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import Callback
from scikitplot.metrics import plot_confusion_matrix, plot_roc

from matplotlib import pyplot

class PerformanceVisualizationCallback(Callback):
  def __init__(self, model, validation_data, image_dir):
    super().__init__()
    self.model = model
    self.validation_data = validation_data

    os.makedirs(image_dir, exist_ok=True)
    self.image_dir = image_dir
  
  def on_train_batch_end(self, batch, logs = None):
      batch_end_loss.append(logs['loss'])
      batch_end_accu.append(logs['accuracy'])
  
  def on_test_batch_end(self, batch, logs = None):
      batch_end_loss_test.append(logs['loss'])
      batch_end_accu_test.append(logs['accuracy'])

 # def on_epoch_end(self,epoch,logs={}):
 #   y_pred = np.asarray(self.model.predict(self.validation_data[0]))
 #   y_true = self.validation_data[1]
 #   y_pred_class = np.argmax(y_pred, axis=1)
 #   y_true_class = np.argmax(y_true,axis=1)

 #   fig, ax = pyplot.subplots(figsize=(16,13))
 #   plot_confusion_matrix(y_true_class, y_pred_class, ax=ax)
 #   fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

    #fig, ax = pyplot.subplots(figsize=(16,13))
    #plot_roc(y_true, y_pred, ax=ax)
    #fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

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
model.compile(loss ='categorical_crossentropy',optimizer=opt,metrics=['accuracy',cpu_usage,'mae'])

model.summary()
keras.utils.plot_model(model,"model-cnn-spectrogram.png",show_shapes=True)

batch_end_loss = list()
batch_end_accu = list()
batch_end_loss_test = list()
batch_end_accu_test = list()

validation_data = x_val, y_val
performance_cbk = PerformanceVisualizationCallback( model=model, validation_data=validation_data, image_dir='performance_visualizations')

if not (os.path.exists('spec_model.h5')):
    performance_cbk = PerformanceVisualizationCallback( model=model, validation_data=validation_data, image_dir='performance_visualizations')
    #history=model.fit(x_train,y_train,epochs=25,batch_size=10,verbose=1,validation_data=validation_data)
    history = model.fit(x_train,y_train,epochs=25,batch_size=10,verbose=1,validation_data=validation_data,callbacks=[performance_cbk])
    print(len(batch_end_loss))
    print(len(batch_end_accu))

    model.save('spec_model.h5')
    eval_history =  model.evaluate(x_val,y_val)
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['mae'])
    pyplot.show()
else:
    loaded_model = keras.models.load_model("spec_model.h5", custom_objects = {'cpu_usage': cpu_usage})
    eval_history = loaded_model.evaluate(x_val,y_val,callbacks = [performance_cbk])
    print(eval_history)
    
    y_pred = np.asarray(loaded_model.predict(validation_data[0]))
    y_true = validation_data[1]
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true,axis=1)

    fig, ax = pyplot.subplots(figsize=(16,13))
    plot_confusion_matrix(y_true_class, y_pred_class, ax=ax)
    fig.savefig(os.path.join('performance_visualizations', f'confusion_matrix'))

