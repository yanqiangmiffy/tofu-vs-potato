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

# Here's our 6 categories that we have to classify.
class_names = ['potato', 'tofu']
class_names_label = {'potato': 1,
                     'tofu': 0,
                     }
nb_classes = 2


def load_data(data_dir, size=(150, 150), is_train=False):
    """
    加载图片数据集
    """
    images = []
    labels = []
    if is_train:
        label_df = pd.read_csv('data/train.csv', header=None)
        label_df.columns = ['id', 'label']
        labels = [str(i) for i in label_df['label'].values.tolist()]
    filenames = os.listdir(data_dir)
    filenames.sort(key=lambda x: int(x[:-4]))
    print(filenames)
    for img_name in tqdm(os.listdir(data_dir)):
        img_path = data_dir + "/" + img_name
        curr_img = cv2.imread(img_path)
        curr_img = cv2.resize(curr_img, size)
        images.append(curr_img)
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    return images, labels


train_images, train_labels = load_data('data/train', is_train=True)
test_images, test_labels = load_data('data/test', is_train=False)

print(test_images.shape[-1])
test_size = test_images.shape[0]
test_df = pd.DataFrame({
    'id': [i for i in range(test_size)]
})
train_images = train_images / 255.0
test_images = test_images / 255.0

index = np.random.randint(train_images.shape[0])
plt.figure()
plt.imshow(train_images[index])
plt.grid(False)
plt.title('Image #{} : '.format(index) + class_names[train_labels[index]])
plt.show()

fig = plt.figure(figsize=(10, 10))
fig.suptitle("Some examples of images of the dataset", fontsize=16)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # the nn will learn the good filter to use
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split=0.2)

fig = plt.figure(figsize=(10, 5))
plt.subplot(221)
plt.plot(history.history['acc'], 'bo--', label="acc")
plt.plot(history.history['val_acc'], 'ro--', label="val_acc")
plt.title("train_acc vs val_acc")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend()

plt.subplot(222)
plt.plot(history.history['loss'], 'bo--', label="loss")
plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")

plt.legend()
plt.show()

predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)

print(pred_labels)
test_df['label'] = pred_labels
test_df[['id', 'label']].to_csv('cnn.csv', index=None, header=False)
