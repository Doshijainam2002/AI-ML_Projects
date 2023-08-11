import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense,Flatten
from keras import Model 
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import requests
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Build the Model to train the Data
base_model = MobileNet(input_shape=(224,224,3), include_top=False) #Weights

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x =Dense(units=7,activation='softmax')(x)

#Create the Model
model = Model(base_model.input,x)

#All the layers of model
model.summary()

#Compile the Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Data Augmentation 
train_datagen = ImageDataGenerator(
    zoom_range = 0.2,
    shear_range=0.2,
    horizontal_flip=True,
    rescale=1./255
    
)

train_data = train_datagen.flow_from_directory(directory="/Users/jainamdoshi/Desktop/Datasets/train",target_size=(244,244),batch_size=32)

val_datagen = ImageDataGenerator(rescale=1/255)
val_data = val_datagen.flow_from_directory(directory="/Users/jainamdoshi/Desktop/Datasets/train",batch_size=32,target_size=(244,244))


#Having Early Stopping and model check point

from keras.callbacks import ModelCheckpoint, EarlyStopping

#Early Stopping
es = EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=5,verbose=1,mode='auto')

#Model check point
mc = ModelCheckpoint(filepath="best_model.h5",monitor='val_accuracy',verbose=1,save_best_only=True,mode='auto')

#Putting callback in a list
call_back = [es,mc]

hist = model.fit_generator(train_data,steps_per_epoch=10,epochs=30, validation_data=val_data,validation_steps=8,callbacks=[es,mc])

from keras.models import load_model
h = hist.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'],c="red")
plt.title("acc bs v-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'],c="red")
plt.title("loss vs v-loss")
plt.show()
