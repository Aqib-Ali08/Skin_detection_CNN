#importing required libraries.........->

import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from keras import layers
from keras import Sequential

#importing dataset.......->

d_train = pathlib.Path("C:\\Users\\Antu Sanbui\\Desktop\\project\\Skin_detection_CNN\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train")
d_test = pathlib.Path("C:\\Users\\Antu Sanbui\\Desktop\\project\\Skin_detection_CNN\\Skin cancer ISIC The International Skin Imaging Collaboration\\Test")

#showing data details........->

print("Training data count :",len(list(d_train.glob('*/*.jpg'))))
print("Testing data count :",len(list(d_test.glob('*/*.jpg'))))

#creating dataset using 80% data for training and 20% data for testing.........->

batch_size = 32
#train dataset->
train_ds = tf.keras.preprocessing.image_dataset_from_directory(d_train,batch_size=batch_size,image_size=(180,180),label_mode='categorical',seed=123,subset="training",validation_split=0.2)
#validation test dataset->
val_ds =tf.keras.preprocessing.image_dataset_from_directory(d_train,batch_size=batch_size,image_size=(180,180),label_mode='categorical',seed=123,subset="validation",validation_split=0.2)

#All the classes of skin cancer.
class_names = train_ds.class_names
print("All Available Classes: \n",class_names)

#collecting all image paths from all classes and storing in a dictionary

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

files_path_dictionary = {}

for c in class_names:
    files_path_dictionary[c] = list(map(lambda x:str(d_train)+'/'+c+'/'+x,os.listdir(str(d_train)+'/'+c)))

#plotting images

plt.figure(figsize=(15,15))
index = 0
for c in class_names:
    path_list = files_path_dictionary[c][:1]
    index += 1
    plt.subplot(3,3,index)
    plt.imshow(load_img(path_list[-1],target_size=(180,180)))
    plt.title(c)
    plt.show()


#After each epoch the loaded images will be kept as cache for the next epoch

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#creating the model

img_shape = (180,180,3)

model = Sequential()     

#First Convulation Layer
model.add(layers.Rescaling(1./255, input_shape=img_shape))
model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))

#Second Convulation Layer
model.add(layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))

#Third Convulation Layer
model.add(layers.Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())   

#Dense Layer
model.add(layers.Dense(512,activation='relu'))
#custom input!
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))

#Dense Layer with softmax activation function.
#Softmax is an activation function that scales numbers/logits into probabilities.
model.add(layers.Dense(len(class_names),activation='softmax'))

model.compile(optimizer='Adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

print(model.summary())

#1st model training

epochs = 20
history = model.fit(train_ds, 
                    validation_data = val_ds, 
                    epochs = epochs)
print(history)

#plotting the graph

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()