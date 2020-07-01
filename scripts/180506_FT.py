# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:10:02 2018

@author: Dennis
"""
#%%

from tensorflow.python.client import device_lib


#%%
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#%%
from __future__ import print_function, division

from keras.backend.tensorflow_backend import set_session  

config = tf.ConfigProto()  
config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction=0.7 
set_session(tf.Session(config=config)) 

 
#%%
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import load_model
from keras.models import save_model
from keras import regularizers

#%%

#from imagenet_utils
from keras.preprocessing import image
from PIL import Image
from keras import backend as K
def preprocess_input_new(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    #x = image.img_to_array(x)
    #x = np.expand_dims(x, axis=0)
    x = x.astype(K.floatx())
    x /= 127.5
    x -= 1.
    return x
    
#%%
def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

#%%
def plot_training(history):
  acc = history.history['categorical_accuracy']
  val_acc = history.history['val_categorical_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'b-', label='training accuracy')
  plt.plot(epochs, val_acc, 'r-', label='validation accuracy')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  

  plt.figure()
  plt.plot(epochs, loss, 'b-', label='training loss')
  plt.plot(epochs, val_loss, 'r-', label='validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()

#%%
  
def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

#%%
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers[:TL_freeze_layers]:
     layer.trainable = False
  for layer in base_model.layers[TL_freeze_layers:]:
     layer.trainable = True
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
  


#%%
def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#%%
  
"""Use transfer learning and fine-tuning to train a network on a new dataset"""

#setup datasets and variables

train_data_dir = 'D:\\Melanoma Class\\ISIC datasets\\train_dir'
test_data_dir = 'D:\\Melanoma Class\\ISIC datasets\\val_dir'


nb_train_samples = get_nb_files('D:\\Melanoma Class\\ISIC datasets\\train_dir')
nb_classes = len(glob.glob('D:\\Melanoma Class\\ISIC datasets\\train_dir' + "/*"))
nb_val_samples = get_nb_files('D:\\Melanoma Class\\ISIC datasets\\val_dir')
nb_epoch = 3
batch_size = 32
    
IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
TL_freeze_layers = 200
NB_IV3_LAYERS_TO_FREEZE = 172


#%%
#initialize InceptionV3 model without top layer

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape = (IM_WIDTH, IM_HEIGHT, 3)) 


#%%

model = add_new_last_layer(base_model, nb_classes)
#%%

setup_to_transfer_learn(model, base_model)
   

#%%

len(base_model.layers)

#%%
len(model.layers)

  
#%%
model.summary()
#%%


#initialize test and training sets

train_datagen =  ImageDataGenerator(
    rescale=1./255, 
    
    fill_mode="nearest",
    
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
  )

test_datagen = ImageDataGenerator(
    rescale=1./255,
    
    fill_mode="nearest",
    
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
  )

train_generator = train_datagen.flow_from_directory(
    'D:\\Melanoma Class\\ISIC datasets\\train_dir',
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    
  )

validation_generator = test_datagen.flow_from_directory(
    'D:\\Melanoma Class\\ISIC datasets\\val_dir',
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    
  )


#%%

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger

file_path_tl = '180510_best_weights_tl.h5'

checkpoint_tl = ModelCheckpoint(file_path_tl, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1 )
early_tl = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.01, patience=5, verbose=1, mode='auto')
#tboard = Tensorboard(log_dir='./logs', histogram_freq=5, batch_size=32, write_graph=False, write_grads=False, write_images=False)
csvlog_tl = CSVLogger('180510_training_tl.log')


#%%
#transfer learning

history_tl = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=nb_val_samples/batch_size,
    class_weight='auto',
    callbacks = [checkpoint_tl, early_tl, csvlog_tl]
)



#%%
plot_training(history_tl)


#%%

import h5py

model.save('180510_tl.h5')


#%%
setup_to_finetune(model)


#%%
model.summary()

#%%
file_path_ft = '180510_best_weights_ft.h5'

checkpoint_ft = ModelCheckpoint(file_path_ft, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1 )
early_ft = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.005, patience=4, verbose=1, mode='auto')
#tboard = Tensorboard(log_dir='./logs', histogram_freq=5, batch_size=32, write_graph=False, write_grads=False, write_images=False)
csvlog_ft = CSVLogger('180510_training_ft.log')

#%%

#fine tuning

history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=nb_val_samples/batch_size,
    class_weight='auto',
    callbacks = [checkpoint_ft, early_ft, csvlog_ft]
    )


#%%
plot_training(history_ft)

#%%
model.save('180510_ft.h5')

