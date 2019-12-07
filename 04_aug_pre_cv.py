#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: Keras+ResNet+TTA.py 
@time: 2019-12-07 02:27
@description:https://www.kaggle.com/bharatsingh213/keras-resnet-tta#Import-Libraries
"""
import numpy as np
import pandas as pd
import os
import cv2
import PIL
import gc
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
from tqdm import tqdm
from math import ceil
import math
import sys
import gc

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from keras.activations import softmax
from keras.activations import elu
from keras.activations import relu
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm
import ipykernel

gc.enable()
SEED = 7
np.random.seed(SEED)
set_random_seed(SEED)
IMG_DIM = 299  # 224 399 #
BATCH_SIZE = 12
CHANNEL_SIZE = 3
NUM_EPOCHS = 20

FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_CLASSES = 2

train_jpg = pd.read_csv('data/train.csv', names=['id', 'label'])
train_jpg['id'] = train_jpg['id'].apply(lambda x: str(x) + '.jpg')
train_jpg['label'] = train_jpg['label'].astype(str)
train_jpg = pd.get_dummies(data=train_jpg, columns=['label'])
y_cols = [col for col in train_jpg.columns if col not in ['id']]


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(dataframe=train_jpg,
                                                    directory="data/train/",
                                                    x_col="id",
                                                    y_col=y_cols,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="raw",
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    # subset='training',
                                                    shuffle=True,
                                                    seed=SEED,
                                                    )

eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
# Reducing the Learning Rate if result is not improving.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',
                              verbose=1)

NUB_TRAIN_STEPS = train_generator.n // train_generator.batch_size


def create_resnet(img_dim, CHANNEL, n_class):
    input_tensor = Input(shape=(img_dim, img_dim, CHANNEL))
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    # base_model.load_weights('../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(n_class, activation='softmax', name="Output_Layer")(x)
    model_resnet = Model(input_tensor, output_layer)

    return model_resnet


model_resnet = create_resnet(IMG_DIM, CHANNEL_SIZE, NUM_CLASSES)

for layers in model_resnet.layers:
    layers.trainable = True

lr = 1e-3
optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)  # Adam(lr=lr, decay=0.01)
model_resnet.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model_resnet.summary()
gc.collect()

history = model_resnet.fit_generator(generator=train_generator,
                                     steps_per_epoch=NUB_TRAIN_STEPS,
                                     epochs=NUM_EPOCHS,
                                     # shuffle=True,
                                     callbacks=[eraly_stop, reduce_lr],
                                     verbose=1)
gc.collect()

# history.history.keys()
#
# accu = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# plt.plot(accu, label="Accuracy")
# plt.plot(val_acc)
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(['Acc', 'val_acc'])
# plt.plot(np.argmax(history.history["val_acc"]), np.max(history.history["val_acc"]), marker="x", color="r",
#          label="best model")
# plt.show()

num_test_img = len(os.listdir("data/test"))
test_filenames = [str(i) + '.jpg' for i in range(num_test_img)]

df_test = pd.DataFrame({
    'filename': test_filenames
})

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  # validation_split=0.2,
                                  horizontal_flip=True)

test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
                                                  directory="data/test",
                                                  x_col="id",
                                                  target_size=(IMG_DIM, IMG_DIM),
                                                  batch_size=1,
                                                  shuffle=False,
                                                  class_mode=None,
                                                  seed=SEED)
# del df_test
print(df_test.shape[0])
# del train_datagen
# del traabsin_generator
gc.collect()

tta_steps = 5
preds_tta = []
for i in tqdm(range(tta_steps)):
    test_generator.reset()
    preds = model_resnet.predict_generator(generator=test_generator, steps=ceil(df_test.shape[0]))
    #     print('Before ', preds.shape)
    preds_tta.append(preds)
#     print(i,  len(preds_tta))


final_pred = np.mean(preds_tta, axis=0)
predicted_class_indices = np.argmax(final_pred, axis=1)
len(predicted_class_indices)

results = pd.DataFrame({"id": [i for i in range(len(df_test))], "label": predicted_class_indices})
# results.id_code = results.id_code.apply(lambda x: x[:-4])  # results.head()
results.to_csv("submission.csv", index=False)

results['label'].value_counts().plot(kind='bar')
plt.title('Test Samples Per Class')
plt.show()

print(results['label'].value_counts())
