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



d_train = pathlib.Path("C:\\Users\\Antu Sanbui\\Desktop\\project\\Skin_detection_CNN\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train")
d_test = pathlib.Path("C:\\Users\\Antu Sanbui\\Desktop\\project\\Skin_detection_CNN\\Skin cancer ISIC The International Skin Imaging Collaboration\\Test")



print("Training data count :",len(list(d_train.glob('*/*.jpg'))))
print("Testing data count :",len(list(d_test.glob('*/*.jpg'))))



batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(d_train,batch_size=batch_size,image_size=(180,180),label_mode='categorical',seed=123,subset="training",validation_split=0.2)

val_ds =tf.keras.preprocessing.image_dataset_from_directory(d_train,batch_size=batch_size,image_size=(180,180),label_mode='categorical',seed=123,subset="validation",validation_split=0.2)

class_names = train_ds.class_names
print("All Available Classes: \n",class_names)



import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

files_path_dictionary = {}

for c in class_names:
    files_path_dictionary[c] = list(map(lambda x:str(d_train)+'/'+c+'/'+x,os.listdir(str(d_train)+'/'+c)))


plt.figure(figsize=(15,15))
index = 0
for c in class_names:
    path_list = files_path_dictionary[c][:1]
    index += 1
    plt.subplot(3,3,index)
    plt.imshow(load_img(path_list[-1],target_size=(180,180)))
    plt.title(c)
    plt.show()


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


img_shape = (180,180,3)

model = Sequential()     

model.add(layers.Rescaling(1./255, input_shape=img_shape))
model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())   

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))

model.add(layers.Dense(len(class_names),activation='softmax'))

model.compile(optimizer='Adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

print(model.summary())


epochs = 20
history = model.fit(train_ds, 
                    validation_data = val_ds, 
                    epochs = epochs)
print(history)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

print("Epoch-wise Training and Validation Metrics:")
print("Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss")
print("-" * 60)
for epoch, (train_acc, val_acc_val, train_loss, val_loss_val) in enumerate(zip(acc, val_acc, loss, val_loss), 1):
    print(f"{epoch:5d} | {train_acc:18.4f} | {val_acc_val:20.4f} | {train_loss:14.4f} | {val_loss_val:16.4f}")

#rescale = tf.keras.Sequential([
    #layers.Rescaling(1./255, input_shape=img_shape)  # Correctly defines the rescaling layer
#])

data_augmentation = tf.keras.Sequential([
    layers.Input(shape=(180, 180, 3)),  
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

rescale = tf.keras.Sequential([
    layers.Rescaling(1./255)  
])

model2 = Sequential()

model2.add(layers.Input(shape=(180, 180, 3)))  
model2.add(data_augmentation)                 
model2.add(rescale)                           

model2.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model2.add(layers.MaxPool2D(pool_size=(2, 2)))

model2.add(layers.Dropout(0.25))

model2.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model2.add(layers.MaxPool2D(pool_size=(2, 2)))

model2.add(layers.Dropout(0.25))

model2.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(layers.MaxPool2D(pool_size=(2, 2)))

model2.add(layers.Flatten())

model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(128, activation='relu'))

model2.add(layers.Dropout(0.50))

model2.add(layers.Dense(len(class_names), activation='softmax'))

model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model2.summary())


epochs = 20
history = model2.fit(train_ds,epochs=epochs,validation_data=val_ds,verbose=1)
print(history)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

print("Epoch-wise Training and Validation Metrics:")
print("Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss")
print("-" * 60)

for epoch, (train_acc, val_acc_val, train_loss, val_loss_val) in enumerate(zip(acc, val_acc, loss, val_loss), start=1):
    print(f"{epoch:5d} | {train_acc:18.4f} | {val_acc_val:20.4f} | {train_loss:14.4f} | {val_loss_val:16.4f}")

import pandas as pd

data = {
    "Epoch": list(range(1, len(acc) + 1)),
    "Training Accuracy": acc,
    "Validation Accuracy": val_acc,
    "Training Loss": loss,
    "Validation Loss": val_loss
}

df = pd.DataFrame(data)
df.to_csv("training_validation_metrics.csv", index=False)
print("\nMetrics saved to 'training_validation_metrics.csv'")

def class_distribution_count(directory):
    
    count= []
    for path in pathlib.Path(directory).iterdir():
        if path.is_dir():
            count.append(len([name for name in os.listdir(path)
                               if os.path.isfile(os.path.join(path, name))]))
    
    
    sub_directory = [name for name in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory, name))]
    

    return pd.DataFrame(list(zip(sub_directory,count)),columns =['Class', 'No. of Image'])

df = class_distribution_count("C:\\Users\\Antu Sanbui\\Desktop\\project\\Skin_detection_CNN\\Skin cancer ISIC The International Skin Imaging Collaboration\\Train")
print(df)
