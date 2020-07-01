# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:37:04 2018

@author: Dennis
"""

#%%
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import load_model
from keras.models import save_model

#%%
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


#%%

model1=load_model('180505_tl.h5')

#%%

model1.summary()

#%%

from PIL import Image
import requests
from io import BytesIO
from keras.preprocessing import image

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  #x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]



def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("benign", "malignant")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


#%%
import numpy as np
from keras.preprocessing import image
from IPython.display import display, Image
from PIL import Image

#%%
IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
target_size= (299, 299)

test_image = image.load_img('G:\ISIC Melanoma Dataset\ISIC-images malignant\Dermoscopedia (CC-BY)\ISIC_0024205.jpg', 
                            target_size = (IM_WIDTH, IM_HEIGHT))

preds=predict(model1, test_image, target_size)

plot_preds(test_image, preds)


#%%
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
#test_image = preprocess_input(test_image)
result = model1.predict(test_image)

result

#%%
target_size= (299, 299)
response = requests.get('https://www.riversideonline.com/xmlParse/source/images/slideshow/ds00575-d-diameter.jpg')
img = Image.open(BytesIO(response.content))
preds = predict(model1, img, target_size)
plot_preds(img, preds)

#%%

test_data_dir = 'D:\\Melanoma Class\\HAM datasets\\datasets\\val_dir'
batch_size = 16
IM_WIDTH, IM_HEIGHT = 299, 299

test_datagen = ImageDataGenerator(
    rescale=1./255,

  )

validation_generator = test_datagen.flow_from_directory(
    'D:\\Melanoma Class\\HAM datasets\\datasets\\val_dir',
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=False,
    class_mode=None
  )  

#%%
proba = model1.predict_generator(validation_generator, verbose=1)

#%%
len(proba)

#%%
validation_generator.class_indices

#%%
y_test=validation_generator.classes

#%%
y_pred = proba[:,1]
y_pred
            
#%%
y_pred_int = np.rint(proba[:,1])

#%%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
#%%
#generate AUC ROC

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()