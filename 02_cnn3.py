#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_cnn.py 
@time: 2019-11-30 00:30
@description:
"""
import pandas as pd
import numpy as np  # linear algebra
import os  # used for loading the data
from sklearn.metrics import confusion_matrix  # confusion matrix to carry out error analysis
import seaborn as sn  # heatmap
from sklearn.utils import shuffle  # shuffle the data
import matplotlib.pyplot as plt  # 2D plotting library
import cv2  # image processing library
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical

# Here's our 6 categories that we have to classify.
class_names = ['potato', 'tofu']
class_names_label = {'potato': 1,
                     'tofu': 0,
                     }
nb_classes = 2


def load_data(data_dir, size=(300, 300), is_train=False):
    """
    加载图片数据集
    """
    images = []
    labels = []
    if is_train:
        label_df = pd.read_csv('data/train.csv', header=None)
        label_df.columns = ['id', 'label']
        labels = [str(i) for i in label_df['label'].values.tolist()]
    for img_name in tqdm(os.listdir(data_dir)):
        img_path = data_dir + "/" + img_name
        curr_img = cv2.imread(img_path)
        curr_img = cv2.resize(curr_img, size)
        images.append(curr_img)
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    return images, labels


train_images, train_labels = load_data('data/train', is_train=True)
train_labels = to_categorical(train_labels,2)
test_images, test_labels = load_data('data/test', is_train=False)

print(test_images.shape[-1])
test_size = test_images.shape[0]
test_df = pd.DataFrame({
    'id': [i for i in range(test_size)]
})
train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(300,300,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split=0.2)

predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)

print(pred_labels)
test_df['label'] = pred_labels
test_df[['id', 'label']].to_csv('cnn.csv', index=None, header=False)
