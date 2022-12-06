import numpy as np
import os
import psutil
import pickle

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import Callback
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.metrics import classification_report

from matplotlib import pyplot

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

test_set_path = 'spectrograms-testing/'

x_test = []
y_test = []

dirs = os.listdir(test_set_path)

for audio_class in dirs:
    files = os.listdir(test_set_path+audio_class)
    for f in files:
        
        y_test.append(classes[audio_class])
        
        file_path = test_set_path+audio_class+'/'+f
        with open(file_path,'rb') as fspec:
            #print(file_path)
            spectro = pickle.load(fspec)
            data_item = np.empty(list(spectro.shape)+[1])
            data_item[:,:,0] = spectro
            x_test.append(data_item)

y_test =np.asarray(y_test)

print(y_test.shape)

y_test = keras.utils.to_categorical(y_test,num_classes=12)

x_test = np.stack(x_test)

testing_data = x_test,y_test

loaded_model = keras.models.load_model("spec_model.h5", custom_objects = {'cpu_usage': cpu_usage})
#eval_history = loaded_model.evaluate(x_val,y_val,callbacks = [performance_cbk])
#print(eval_history)
    
y_pred = np.asarray(loaded_model.predict(testing_data[0]))
y_true = testing_data[1]
y_pred_class = np.argmax(y_pred, axis=1)
y_true_class = np.argmax(y_true, axis=1)

class_label = list(classes.keys())

y_pred_labeled = []
y_true_labeled = []

for i in y_pred_class:
    y_pred_labeled.append(class_label[i])
for i in y_true_class:
    y_true_labeled.append(class_label[i])

fig, ax = pyplot.subplots(figsize=(16,13))
plot_confusion_matrix(y_true_labeled, y_pred_labeled, ax=ax, labels=class_label)
fig.savefig(os.path.join('performance_visualizations_test', f'confusion_matrix'))

print(classification_report(y_true_class, y_pred_class, target_names=classes))
