import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout, \
    Lambda, Convolution2D, \
    MaxPooling2D, Flatten, BatchNormalization, \
    Conv2D, AveragePooling2D, Activation, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import layers, models, optimizers
from efficientnet.tfkeras import EfficientNetB7
from utils import flip, color, zoom, rotate
import numpy as np
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# use multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

num_classes = 2
image_size = 128
batch_size = 64
learn_rate = 0.0001
pretrain_epochs = 20
train_epochs = 30

use_data_aug = True

root_path = os.getcwd()
logs = os.path.join(root_path, 'kaggle_logs_v3')
train_files = os.listdir('train')
kaggle_train_files = os.listdir('kaggle_train')
val_files = os.listdir('val')
test_files = os.listdir('test')
if not os.path.exists(logs):
    os.mkdir(logs)

train_root = ('train')
kaggle_train_root = ('kaggle_train')
val_root = ('val')
test_root = ('test')
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }}

train_image_names = []
train_labels = []

kaggle_train_names = []
kaggle_train_labels = []

val_image_names = []
val_labels = []

for i in train_files:
    train_image_names.append(i)
    if i.split('_')[0] == 'cat':
        train_labels.append(0)
    if i.split('_')[0] == 'dog':
        train_labels.append(1)

# get kaggle train dataset label
for i in kaggle_train_files:
    kaggle_train_names.append(i)
    if i.split('.')[0] == 'cat':
        kaggle_train_labels.append(0)
    if i.split('.')[0] == 'dog':
        kaggle_train_labels.append(1)

for i in val_files:
    val_image_names.append(i)
    if i.split('_')[0] == 'cat':
        val_labels.append(0)
    if i.split('_')[0] == 'dog':
        val_labels.append(1)

a = []
for i in range(0, 2000):
    g = str(i) + '.jpg'
    a.append(g)
train_datasets_x = tf.data.Dataset.from_tensor_slices(train_image_names)
train_datasets_y = tf.data.Dataset.from_tensor_slices(train_labels)
kaggle_train_datasets_x = tf.data.Dataset.from_tensor_slices(kaggle_train_names)
kaggle_train_datasets_y = tf.data.Dataset.from_tensor_slices(kaggle_train_labels)

val_datasets = tf.data.Dataset.from_tensor_slices((val_image_names, val_labels))

test_datasets = tf.data.Dataset.from_tensor_slices(a)


def kaggle_train_parse(image_names):
    image_names = tf.cast(image_names, tf.string)
    train_file_path = kaggle_train_root + os.sep + image_names
    image = tf.io.read_file(train_file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.divide(image, 255.0)
    return image


def train_parse(image_names):
    image_names = tf.cast(image_names, tf.string)
    train_file_path = train_root + os.sep + image_names
    image = tf.io.read_file(train_file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.divide(image, 255.0)
    return image


def val_parse(image_names, label):
    image_names = tf.cast(image_names, tf.string)
    val_file_path = val_root + os.sep + image_names
    label = tf.cast(label, tf.uint8)
    image = tf.io.read_file(val_file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.divide(image, 255.0)
    return image, label


def test_parse(image_names):
    image_names = tf.cast(image_names, tf.string)
    test_file_path = test_root + os.sep + image_names

    image = tf.io.read_file(test_file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.divide(image, 255.0)
    return image


# define model
def EfficientNet(input_size=(image_size, image_size, 3)):
    '''
    # fine tune EfficientNet
    :param input_size: input image size ,int8
    :return: keras model
    '''
    base_model = EfficientNetB7(include_top=False, weights='imagenet', pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = layers.Dropout(0.2, name='top_dropout')(x)
    predictions = layers.Dense(1,
                               activation='sigmoid',
                               kernel_initializer=DENSE_KERNEL_INITIALIZER,
                               name='probs')(x)
    model = Model(base_model.input, predictions, name='sigmoidmodel')
    input_size = Input(input_size)
    output = model(input_size)
    model = Model(input_size, output)
    return model


train_datasets_x = train_datasets_x.map(train_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
kaggle_train_datasets_x = kaggle_train_datasets_x.map(kaggle_train_parse,
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
augmentations = [flip, color, zoom, rotate]

for g in augmentations:
    train_dataset_x = train_datasets_x.map \
        (lambda x: tf.cond(tf.random.uniform([], 0, 1) > tf.constant(0.75), \
                           lambda: g(x), lambda: x), \
         num_parallel_calls=tf.data.experimental.AUTOTUNE
         )
    kaggle_train_datasets_x = kaggle_train_datasets_x.map(
        lambda x: tf.cond(tf.random.uniform([], 0, 1) > tf.constant(0.75), \
                          lambda: g(x), lambda: x), \
        num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

train_datasets_x = train_datasets_x.map(lambda x: tf.clip_by_value(x, 0, 1), tf.data.experimental.AUTOTUNE)
kaggle_train_datasets_x = kaggle_train_datasets_x.map(lambda x: tf.clip_by_value(x, 0, 1),
                                                      tf.data.experimental.AUTOTUNE)

train_datasets = tf.data.Dataset.zip((train_datasets_x, train_datasets_y))
kaggle_train_datasets = tf.data.Dataset.zip((kaggle_train_datasets_x, kaggle_train_datasets_y))

train_datasets = train_datasets.shuffle(20000).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
kaggle_train_datasets = kaggle_train_datasets.shuffle(25000).batch(batch_size).repeat().prefetch(
    tf.data.experimental.AUTOTUNE)
val_datasets = val_datasets.map(val_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(2000).batch(
    batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)


def lr_schedule(epoch):
    """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
        epoch (int): The number of epochs
        # Returns
        lr (float32): learning rate
        """
    lr = 1e-4
    if epoch > 40:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 0.5
    elif epoch > 20:
        lr *= 0.5
    elif epoch > 10:
        lr *= 0.5
    print('Learning rate: ', lr)
    return lr


checkpoint = ModelCheckpoint(os.path.join(logs, 'B7ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.ckt'),
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)

tensorborad = tf.keras.callbacks.TensorBoard(log_dir=logs)
callbacks = [checkpoint, tensorborad]

# ---------------------------------
# pretrain with kaggle cat vs dog data
# --------------------------------


print('pretrain with kaggle cat vs dog datasets.')

with strategy.scope():
    model = EfficientNet()

    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.RMSprop(lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
    model.fit(kaggle_train_datasets,

              validation_data=val_datasets,
              epochs=pretrain_epochs,
              steps_per_epoch=int(20000 / batch_size),
              validation_steps=int(2000 / batch_size),
              callbacks=callbacks)

    if True:
        print('continue train with competition datasets.')
        for layer in model.layers:
            layer.trainable = True

        model.compile(loss='binary_crossentropy',
                      optimizer=tf.optimizers.RMSprop(lr_schedule(0)),
                      metrics=['accuracy'])
        model.summary()
        model.fit(train_datasets,

                  validation_data=val_datasets,
                  epochs=40,
                  initial_epoch=pretrain_epochs,
                  steps_per_epoch=int(20000 / batch_size),
                  validation_steps=int(2000 / batch_size),
                  callbacks=callbacks)

model = EfficientNet()
model.load_weights(os.path.join(logs, 'final_model.h5'))
test_datasets = test_datasets.map(test_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

results = []
predicts = model.predict(test_datasets)

for r in predicts:
    if r >= 0.5:
        results.append(1)
    if r < 0.5:
        results.append(0)

results = pd.Series(results)
submission = pd.concat([pd.Series(range(0, 2000)), results], axis=1)
submission.to_csv(os.path.join(logs, 'submission.csv'), index=False)


