# -*- coding: utf-8 -*-
"""inceptionUNET

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JXJhhrExrTuUw00j8SzNxKOlvboDzvIx
"""

"""In case you are running a [Docker](https://docs.docker.com/install/) image of [Jupyter Notebook server using TensorFlow's nightly](https://www.tensorflow.org/install/docker#examples_using_cpu-only_images), it is necessary to expose not only the notebook's port, but the TensorBoard's port.

Thus, run the container with the following command:

```
docker run -it -p 8888:8888 -p 6006:6006 \
tensorflow/tensorflow:nightly-py3-jupyter 
```

where the `-p 6006` is the default port of TensorBoard. This will allocate a port for you to run one TensorBoard instance. To have concurrent instances, it is necessary to allocate more ports.

"""



# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from tqdm import tqdm 
import pdb

#from google.colab import drive
#drive.mount('/content/drive/')
#drive.flush_and_unmount()

pdb.set_trace()

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

path = '/data/output'

maskpath ='/data/input'

X_train = np.zeros((len(os.listdir(path)), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
Y_train = np.zeros((len(os.listdir(maskpath)), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))

print('creating input data')
n=0
for name in tqdm(sorted(os.listdir(path))):
    newpath = os.path.join(path,name)
    img=cv2.imread(newpath)
    img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    X_train[n]=img/255
    n+=1

m=0
for name in tqdm(sorted(os.listdir(maskpath))):
    newmaskpath = os.path.join(maskpath,name)
    mpg=cv2.imread(newmaskpath)
    mpg=cv2.resize(mpg,(IMG_WIDTH,IMG_HEIGHT))
    #mpg=np.expand_dims(mpg,axis=-1)
    Y_train[m]=mpg/255
    m+=1


def incepUNET():
  inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

  ##DOWNSAMPLING
  ##layer1

  #A member
  c1A = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(s)
  c1A = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(c1A)
  c1A = tf.keras.layers.BatchNormalization()(c1A)
  c1A = tf.keras.layers.ReLU()(c1A)

  #B member
  c1B = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(s)
  c1B = tf.keras.layers.BatchNormalization()(c1B)
  c1B = tf.keras.layers.ReLU()(c1B)

  #C member
  c1C = tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer='he_normal', padding='same')(s)
  c1C = tf.keras.layers.BatchNormalization()(c1C)
  c1C = tf.keras.layers.ReLU()(c1C)
  c1C = tf.keras.layers.Conv2D(5, (3, 3), kernel_initializer='he_normal', padding='same')(c1C)
  c1C = tf.keras.layers.BatchNormalization()(c1C)
  c1C = tf.keras.layers.ReLU()(c1C)

  #D member
  c1D = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(s)
  c1D = tf.keras.layers.BatchNormalization()(c1D)
  c1D = tf.keras.layers.ReLU()(c1D)
  c1D = tf.keras.layers.Conv2D(5, (5, 5), kernel_initializer='he_normal', padding='same')(c1D)
  c1D = tf.keras.layers.BatchNormalization()(c1D)
  c1D = tf.keras.layers.ReLU()(c1D)

  C1 = tf.keras.layers.concatenate([c1A,c1B,c1C,c1D])

  #A member
  c1A = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C1)
  c1A = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(c1A)
  c1A = tf.keras.layers.BatchNormalization()(c1A)
  c1A = tf.keras.layers.ReLU()(c1A)

  #B member
  c1B = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(C1)
  c1B = tf.keras.layers.BatchNormalization()(c1B)
  c1B = tf.keras.layers.ReLU()(c1B)

  #C member
  c1C = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C1)
  c1C = tf.keras.layers.BatchNormalization()(c1C)
  c1C = tf.keras.layers.ReLU()(c1C)
  c1C = tf.keras.layers.Conv2D(10, (3, 3), kernel_initializer='he_normal', padding='same')(c1C)
  c1C = tf.keras.layers.BatchNormalization()(c1C)
  c1C = tf.keras.layers.ReLU()(c1C)

  #D member
  c1D = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C1)
  c1D = tf.keras.layers.BatchNormalization()(c1D)
  c1D = tf.keras.layers.ReLU()(c1D)
  c1D = tf.keras.layers.Conv2D(10, (5, 5), kernel_initializer='he_normal', padding='same')(c1D)
  c1D = tf.keras.layers.BatchNormalization()(c1D)
  c1D = tf.keras.layers.ReLU()(c1D)

  C1 = tf.keras.layers.concatenate([c1A,c1B,c1C,c1D])

  #A member
  c1A = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C1)
  c1A = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(c1A)
  c1A = tf.keras.layers.BatchNormalization()(c1A)
  c1A = tf.keras.layers.ReLU()(c1A)

  #B member
  c1B = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(C1)
  c1B = tf.keras.layers.BatchNormalization()(c1B)
  c1B = tf.keras.layers.ReLU()(c1B)

  #C member
  c1C = tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer='he_normal', padding='same')(C1)
  c1C = tf.keras.layers.BatchNormalization()(c1C)
  c1C = tf.keras.layers.ReLU()(c1C)
  c1C = tf.keras.layers.Conv2D(5, (3, 3), kernel_initializer='he_normal', padding='same')(c1C)
  c1C = tf.keras.layers.BatchNormalization()(c1C)
  c1C = tf.keras.layers.ReLU()(c1C)

  #D member
  c1D = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C1)
  c1D = tf.keras.layers.BatchNormalization()(c1D)
  c1D = tf.keras.layers.ReLU()(c1D)
  c1D = tf.keras.layers.Conv2D(5, (5, 5), kernel_initializer='he_normal', padding='same')(c1D)
  c1D = tf.keras.layers.BatchNormalization()(c1D)
  c1D = tf.keras.layers.ReLU()(c1D)

  C1 = tf.keras.layers.concatenate([c1A,c1B,c1C,c1D])

  p1 = tf.keras.layers.MaxPooling2D((2, 2))(C1)

  ##layer2

  #A member
  c2A = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(p1)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  c2A = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2A)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  c2A = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c2A)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  #B member
  c2B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(p1)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  c2B = tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer='he_normal', padding='same')(c2B)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  c2B = tf.keras.layers.Conv2D(16, (5, 5), kernel_initializer='he_normal', padding='same')(c2B)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  C2 = tf.keras.layers.concatenate([c2A,c2B])

  #A member
  c2A = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C2)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  c2A = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c2A)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  c2A = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2A)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  #B member
  c2B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C2)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  c2B = tf.keras.layers.Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same')(c2B)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  c2B = tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer='he_normal', padding='same')(c2B)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  C2 = tf.keras.layers.concatenate([c2A,c2B])

  #A member
  c2A = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C2)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  c2A = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2A)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  c2A = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c2A)
  c2A = tf.keras.layers.BatchNormalization()(c2A)
  c2A = tf.keras.layers.ReLU()(c2A)

  #B member
  c2B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C2)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  c2B = tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer='he_normal', padding='same')(c2B)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  c2B = tf.keras.layers.Conv2D(16, (5, 5), kernel_initializer='he_normal', padding='same')(c2B)
  c2B = tf.keras.layers.BatchNormalization()(c2B)
  c2B = tf.keras.layers.ReLU()(c2B)

  C2 = tf.keras.layers.concatenate([c2A,c2B])

  p2 = tf.keras.layers.MaxPooling2D((2, 2))(C2)



  ## layer 3

  #A member
  c3A = tf.keras.layers.Conv2D(256, (1, 1), kernel_initializer='he_normal', padding='same')(p2)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(128, (1, 7), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(64, (7,1), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(32, (1, 7), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A) 

  #B member
  c3B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(p2)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B)

  c3B = tf.keras.layers.Conv2D(32, (1,7), kernel_initializer='he_normal', padding='same')(c3B)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B)

  c3B = tf.keras.layers.Conv2D(20, (1, 7), kernel_initializer='he_normal', padding='same')(c3B)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B) 

  #C member
  c3C = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(p2)
  c3C = tf.keras.layers.BatchNormalization()(c3C)
  c3C = tf.keras.layers.ReLU()(c3C)

  #D
  c3D = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(p2)
  c3D = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(c3D)
  c3D = tf.keras.layers.BatchNormalization()(c3D)
  c3D = tf.keras.layers.ReLU()(c3D)

  C3 = tf.keras.layers.concatenate([c3A,c3B,c3C,c3D])



  #A member
  c3A = tf.keras.layers.Conv2D(512, (1, 1), kernel_initializer='he_normal', padding='same')(C3)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(256, (1, 7), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(128, (7,1), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(64, (1, 7), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A) 

  #B member
  c3B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C3)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B)

  c3B = tf.keras.layers.Conv2D(64, (1,7), kernel_initializer='he_normal', padding='same')(c3B)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B)

  c3B = tf.keras.layers.Conv2D(40, (1, 7), kernel_initializer='he_normal', padding='same')(c3B)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B) 

  #C member
  c3C = tf.keras.layers.Conv2D(12, (1, 1), kernel_initializer='he_normal', padding='same')(C3)
  c3C = tf.keras.layers.BatchNormalization()(c3C)
  c3C = tf.keras.layers.ReLU()(c3C)

  #D member
  c3D = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C3)
  c3D = tf.keras.layers.Conv2D(12, (1, 1), kernel_initializer='he_normal', padding='same')(c3D)
  c3D = tf.keras.layers.BatchNormalization()(c3D)
  c3D = tf.keras.layers.ReLU()(c3D)

  C3 = tf.keras.layers.concatenate([c3A,c3B,c3C,c3D])




  #A member
  c3A = tf.keras.layers.Conv2D(256, (1, 1), kernel_initializer='he_normal', padding='same')(C3)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(128, (1, 7), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(64, (7,1), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A)

  c3A = tf.keras.layers.Conv2D(32, (1, 7), kernel_initializer='he_normal', padding='same')(c3A)
  c3A = tf.keras.layers.BatchNormalization()(c3A)
  c3A = tf.keras.layers.ReLU()(c3A) 

  #B member
  c3B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C3)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B)

  c3B = tf.keras.layers.Conv2D(32, (1,7), kernel_initializer='he_normal', padding='same')(c3B)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B)

  c3B = tf.keras.layers.Conv2D(20, (1, 7), kernel_initializer='he_normal', padding='same')(c3B)
  c3B = tf.keras.layers.BatchNormalization()(c3B)
  c3B = tf.keras.layers.ReLU()(c3B) 

  #C member
  c3C = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(C3)
  c3C = tf.keras.layers.BatchNormalization()(c3C)
  c3C = tf.keras.layers.ReLU()(c3C)

  #D member
  c3D = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C3)
  c3D = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(c3D)
  c3D = tf.keras.layers.BatchNormalization()(c3D)
  c3D = tf.keras.layers.ReLU()(c3D)

  C3 = tf.keras.layers.concatenate([c3A,c3B,c3C,c3D])
  p3 = tf.keras.layers.MaxPooling2D((2, 2))(C3)


  ## layer 4

  #A member
  c4A = tf.keras.layers.Conv2D(28, (1, 1), kernel_initializer='he_normal', padding='same')(p3)
  c4A = tf.keras.layers.BatchNormalization()(c4A)
  c4A = tf.keras.layers.ReLU()(c4A)

  #B member

  c4B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(p3)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  c4B = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4B)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  c4B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(c4B)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  #C member

  c4C = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(p3)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(256, (1, 3), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(128, (3, 1), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  C4 = tf.keras.layers.concatenate([c4A,c4B,c4C])


  #A member
  c4A = tf.keras.layers.Conv2D(56, (1, 1), kernel_initializer='he_normal', padding='same')(C4)
  c4A = tf.keras.layers.BatchNormalization()(c4A)
  c4A = tf.keras.layers.ReLU()(c4A)

  #B member

  c4B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C4)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  c4B = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c4B)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  c4B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(c4B)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  #C member

  c4C = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C4)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(512, (1, 3), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(256, (3, 1), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(72, (3, 3), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  C4 = tf.keras.layers.concatenate([c4A,c4B,c4C])



  #A member
  c4A = tf.keras.layers.Conv2D(28, (1, 1), kernel_initializer='he_normal', padding='same')(C4)
  c4A = tf.keras.layers.BatchNormalization()(c4A)
  c4A = tf.keras.layers.ReLU()(c4A)

  #B member

  c4B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C4)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  c4B = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4B)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  c4B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(c4B)
  c4B = tf.keras.layers.BatchNormalization()(c4B)
  c4B = tf.keras.layers.ReLU()(c4B)

  #C member

  c4C = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C4)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(256, (1, 3), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(128, (3, 1), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  c4C = tf.keras.layers.Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same')(c4C)
  c4C = tf.keras.layers.BatchNormalization()(c4C)
  c4C = tf.keras.layers.ReLU()(c4C)

  C4 = tf.keras.layers.concatenate([c4A,c4B,c4C])
  p4 = tf.keras.layers.MaxPooling2D((2, 2))(C4)

  ##BRIDGE
  ##Layer5
  c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
  c5 = tf.keras.layers.BatchNormalization()(c5)
  c5 = tf.keras.layers.ReLU()(c5)
  c5 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
  c5 = tf.keras.layers.BatchNormalization()(c5)
  c5 = tf.keras.layers.ReLU()(c5)
  c5 = tf.keras.layers.Dropout(0.3)(c5)
  c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
  c5 = tf.keras.layers.BatchNormalization()(c5)
  C5 = tf.keras.layers.ReLU()(c5)

  ##UPSAMPLING
  ##layer 6
  u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
  u6 = tf.keras.layers.concatenate([u6, C4])

  #A member
  c6A = tf.keras.layers.Conv2D(28, (1, 1), kernel_initializer='he_normal', padding='same')(u6)
  c6A = tf.keras.layers.BatchNormalization()(c6A)
  c6A = tf.keras.layers.ReLU()(c6A)

  #B member

  c6B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(u6)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  c6B = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6B)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  c6B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(c6B)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  #C member

  c6C = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(u6)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(256, (1, 3), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(128, (3, 1), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  C6 = tf.keras.layers.concatenate([c6A,c6B,c6C])


  #A member
  c6A = tf.keras.layers.Conv2D(56, (1, 1), kernel_initializer='he_normal', padding='same')(C6)
  c6A = tf.keras.layers.BatchNormalization()(c6A)
  c6A = tf.keras.layers.ReLU()(c6A)

  #B member

  c6B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C6)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  c6B = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c6B)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  c6B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(c6B)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  #C member

  c6C = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C6)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(512, (1, 3), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(256, (3, 1), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(72, (3, 3), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  C6 = tf.keras.layers.concatenate([c6A,c6B,c6C])



  #A member
  c6A = tf.keras.layers.Conv2D(28, (1, 1), kernel_initializer='he_normal', padding='same')(C6)
  c6A = tf.keras.layers.BatchNormalization()(c6A)
  c6A = tf.keras.layers.ReLU()(c6A)

  #B member

  c6B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C6)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  c6B = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6B)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  c6B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(c6B)
  c6B = tf.keras.layers.BatchNormalization()(c6B)
  c6B = tf.keras.layers.ReLU()(c6B)

  #C member

  c6C = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C6)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(256, (1, 3), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(128, (3, 1), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  c6C = tf.keras.layers.Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same')(c6C)
  c6C = tf.keras.layers.BatchNormalization()(c6C)
  c6C = tf.keras.layers.ReLU()(c6C)

  C6 = tf.keras.layers.concatenate([c6A,c6B,c6C])

  ##layer 7
  u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(C6)
  u7 = tf.keras.layers.concatenate([u7, C3])

  #A member
  c7A = tf.keras.layers.Conv2D(256, (1, 1), kernel_initializer='he_normal', padding='same')(u7)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(128, (1, 7), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(64, (7,1), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(32, (1, 7), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A) 

  #B member
  c7B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(u7)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B)

  c7B = tf.keras.layers.Conv2D(32, (1,7), kernel_initializer='he_normal', padding='same')(c7B)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B)

  c7B = tf.keras.layers.Conv2D(20, (1, 7), kernel_initializer='he_normal', padding='same')(c7B)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B) 

  #C member
  c7C = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(u7)
  c7C = tf.keras.layers.BatchNormalization()(c7C)
  c7C = tf.keras.layers.ReLU()(c7C)

  #D
  c7D = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(u7)
  c7D = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(c7D)
  c7D = tf.keras.layers.BatchNormalization()(c7D)
  c7D = tf.keras.layers.ReLU()(c7D)

  C7 = tf.keras.layers.concatenate([c3A,c3B,c3C,c3D])



  #A member
  c7A = tf.keras.layers.Conv2D(512, (1, 1), kernel_initializer='he_normal', padding='same')(C7)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(256, (1, 7), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(128, (7,1), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(64, (1, 7), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A) 

  #B member
  c7B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C7)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B)

  c7B = tf.keras.layers.Conv2D(64, (1,7), kernel_initializer='he_normal', padding='same')(c7B)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B)

  c7B = tf.keras.layers.Conv2D(40, (1, 7), kernel_initializer='he_normal', padding='same')(c7B)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B) 

  #C member
  c7C = tf.keras.layers.Conv2D(12, (1, 1), kernel_initializer='he_normal', padding='same')(C7)
  c7C = tf.keras.layers.BatchNormalization()(c7C)
  c7C = tf.keras.layers.ReLU()(c7C)

  #D
  c7D = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C7)
  c7D = tf.keras.layers.Conv2D(12, (1, 1), kernel_initializer='he_normal', padding='same')(c7D)
  c7D = tf.keras.layers.BatchNormalization()(c7D)
  c7D = tf.keras.layers.ReLU()(c7D)

  C7 = tf.keras.layers.concatenate([c7A,c7B,c7C,c7D])




  #A member
  c7A = tf.keras.layers.Conv2D(256, (1, 1), kernel_initializer='he_normal', padding='same')(C7)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(128, (1, 7), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(64, (7,1), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A)

  c7A = tf.keras.layers.Conv2D(32, (1, 7), kernel_initializer='he_normal', padding='same')(c7A)
  c7A = tf.keras.layers.BatchNormalization()(c7A)
  c7A = tf.keras.layers.ReLU()(c7A) 

  #B member
  c7B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C7)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B)

  c7B = tf.keras.layers.Conv2D(32, (1,7), kernel_initializer='he_normal', padding='same')(c7B)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B)

  c7B = tf.keras.layers.Conv2D(20, (1, 7), kernel_initializer='he_normal', padding='same')(c7B)
  c7B = tf.keras.layers.BatchNormalization()(c7B)
  c7B = tf.keras.layers.ReLU()(c7B) 

  #C member
  c7C = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(C7)
  c7C = tf.keras.layers.BatchNormalization()(c7C)
  c7C = tf.keras.layers.ReLU()(c7C)

  #D member
  c7D = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C7)
  c7D = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(c7D)
  c7D = tf.keras.layers.BatchNormalization()(c7D)
  c7D = tf.keras.layers.ReLU()(c7D)

  C7 = tf.keras.layers.concatenate([c7A,c7B,c7C,c7D])

  ##layer 8
  u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(C7)
  u8 = tf.keras.layers.concatenate([u8, C2])


  #A member
  c8A = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(u8)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  c8A = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8A)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  c8A = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c8A)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  #B member
  c8B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(u8)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  c8B = tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer='he_normal', padding='same')(c8B)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  c8B = tf.keras.layers.Conv2D(16, (5, 5), kernel_initializer='he_normal', padding='same')(c8B)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  C8 = tf.keras.layers.concatenate([c8A,c8B])

  #A member
  c8A = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C8)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  c8A = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c8A)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  c8A = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8A)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  #B member
  c8B = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C8)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  c8B = tf.keras.layers.Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same')(c8B)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  c8B = tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer='he_normal', padding='same')(c8B)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  C8 = tf.keras.layers.concatenate([c8A,c8B])

  #A member
  c8A = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C8)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  c8A = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8A)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  c8A = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c8A)
  c8A = tf.keras.layers.BatchNormalization()(c8A)
  c8A = tf.keras.layers.ReLU()(c8A)

  #B member
  c8B = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C8)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  c8B = tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer='he_normal', padding='same')(c8B)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  c8B = tf.keras.layers.Conv2D(16, (5, 5), kernel_initializer='he_normal', padding='same')(c8B)
  c8B = tf.keras.layers.BatchNormalization()(c8B)
  c8B = tf.keras.layers.ReLU()(c8B)

  C8 = tf.keras.layers.concatenate([c8A,c8B])

  ##layer 9
  u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(C8)
  u9 = tf.keras.layers.concatenate([u9, C1], axis=3)
  #A member
  c9A = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(u9)
  c9A = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(c9A)
  c9A = tf.keras.layers.BatchNormalization()(c9A)
  c9A = tf.keras.layers.ReLU()(c9A)

  #B member
  c9B = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(u9)
  c9B = tf.keras.layers.BatchNormalization()(c9B)
  c9B = tf.keras.layers.ReLU()(c9B)

  #C member
  c9C = tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer='he_normal', padding='same')(u9)
  c9C = tf.keras.layers.BatchNormalization()(c9C)
  c9C = tf.keras.layers.ReLU()(c9C)
  c9C = tf.keras.layers.Conv2D(5, (3, 3), kernel_initializer='he_normal', padding='same')(c9C)
  c9C = tf.keras.layers.BatchNormalization()(c9C)
  c9C = tf.keras.layers.ReLU()(c9C)

  #D member
  c9D = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(u9)
  c9D = tf.keras.layers.BatchNormalization()(c9D)
  c9D = tf.keras.layers.ReLU()(c9D)
  c9D = tf.keras.layers.Conv2D(5, (5, 5), kernel_initializer='he_normal', padding='same')(c9D)
  c9D = tf.keras.layers.BatchNormalization()(c9D)
  c9D = tf.keras.layers.ReLU()(c9D)

  C9 = tf.keras.layers.concatenate([c9A,c9B,c9C,c9D])

  #A member
  c9A = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C9)
  c9A = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(c9A)
  c9A = tf.keras.layers.BatchNormalization()(c9A)
  c9A = tf.keras.layers.ReLU()(c9A)

  #B member
  c9B = tf.keras.layers.Conv2D(6, (1, 1), kernel_initializer='he_normal', padding='same')(C9)
  c9B = tf.keras.layers.BatchNormalization()(c9B)
  c9B = tf.keras.layers.ReLU()(c9B)

  #C member
  c9C = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C9)
  c9C = tf.keras.layers.BatchNormalization()(c9C)
  c9C = tf.keras.layers.ReLU()(c9C)
  c9C = tf.keras.layers.Conv2D(10, (3, 3), kernel_initializer='he_normal', padding='same')(c9C)
  c9C = tf.keras.layers.BatchNormalization()(c9C)
  c9C = tf.keras.layers.ReLU()(c9C)

  #D member
  c9D = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', padding='same')(C9)
  c9D = tf.keras.layers.BatchNormalization()(c9D)
  c9D = tf.keras.layers.ReLU()(c9D)
  c9D = tf.keras.layers.Conv2D(10, (5, 5), kernel_initializer='he_normal', padding='same')(c9D)
  c9D = tf.keras.layers.BatchNormalization()(c9D)
  c9D = tf.keras.layers.ReLU()(c9D)

  C9 = tf.keras.layers.concatenate([c9A,c9B,c9C,c9D])

  #A member
  c9A = tf.keras.layers.MaxPooling2D((2, 2),(1,1),padding='same')(C9)
  c9A = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(c9A)
  c9A = tf.keras.layers.BatchNormalization()(c9A)
  c9A = tf.keras.layers.ReLU()(c9A)

  #B member
  c9B = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer='he_normal', padding='same')(C9)
  c9B = tf.keras.layers.BatchNormalization()(c9B)
  c9B = tf.keras.layers.ReLU()(c9B)

  #C member
  c9C = tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer='he_normal', padding='same')(C9)
  c9C = tf.keras.layers.BatchNormalization()(c9C)
  c9C = tf.keras.layers.ReLU()(c9C)
  c9C = tf.keras.layers.Conv2D(5, (3, 3), kernel_initializer='he_normal', padding='same')(c9C)
  c9C = tf.keras.layers.BatchNormalization()(c9C)
  c9C = tf.keras.layers.ReLU()(c9C)

  #D member
  c9D = tf.keras.layers.Conv2D(64, (1, 1), kernel_initializer='he_normal', padding='same')(C9)
  c9D = tf.keras.layers.BatchNormalization()(c9D)
  c9D = tf.keras.layers.ReLU()(c9D)
  c9D = tf.keras.layers.Conv2D(5, (5, 5), kernel_initializer='he_normal', padding='same')(c9D)
  c9D = tf.keras.layers.BatchNormalization()(c9D)
  c9D = tf.keras.layers.ReLU()(c9D)

  C9 = tf.keras.layers.concatenate([c9A,c9B,c9C,c9D])





  outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(C9)
  
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer='adam', loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()])
  return model

model=incepUNET()

model.summary()

OUTPUT = model.fit(X_train,Y_train,validation_split = 0.1,batch_size=32,epochs=10)


img=model.predict(X_train[2:3],verbose=1)
img=np.array(img,dtype=np.uint8)
img=np.squeeze(img)
cv2.imwrite('img.jpg',img)
plt.imshow(np.squeeze(img))
#plt.imshow(X_train[1])


model.save("/models/InceptionUnet.h5")