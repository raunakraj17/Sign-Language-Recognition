import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Dataset Path
dr = 'Data'
os.listdir(dr)
test_path = dr + '/test/'
train_path = dr +'/train/'

#Processing Data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_train_gen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
image_test_gen = ImageDataGenerator(rescale=1./255)

#Creating Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (310,310,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),input_shape = (310,310,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),input_shape = (310,310,1)))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(96,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(27,activation='softmax'))

#Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#Loading Data
image_shape = (310,310)
train_image_gen = image_train_gen.flow_from_directory(train_path,
                                                    target_size=image_shape,
                                                    color_mode='grayscale',
                                                    batch_size=5,
                                                    class_mode='categorical')
test_image_gen = image_test_gen.flow_from_directory(test_path,
                                                    target_size=image_shape,
                                                    color_mode='grayscale',
                                                    batch_size=2,
                                                    class_mode='categorical',
                                                    shuffle=False)

#Fitting and testing the model

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=3)

results = model.fit(train_image_gen, steps_per_epoch=2569, epochs=10, validation_data=test_image_gen, validation_steps=2134, callbacks=[early_stop])

model.evaluate(test_image_gen)

#Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
model.save_weights('model.h5')
print('Weights saved')