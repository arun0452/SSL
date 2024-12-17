import os
import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from resnet import resnet_v1
from utils import make_rotated_data
from lr_schedule import lr_schedule


def make_ssl_backbone(Train, Test, save_dir, input_shape=(32, 32, 3), n=3, model_name='Restnetv1_SSL_Rotation.keras'):
    (x_rot_train, y_rot_train), (x_rot_test, y_rot_test) = make_rotated_data(Train, Test)

    x_rot_train = x_rot_train.astype('float32') / 255.0
    x_rot_test = x_rot_test.astype('float32') / 255.0

    depth = n * 6 + 2
    resnet_model = resnet_v1(input_shape=input_shape, depth=depth)
    x = Dense(4, activation='softmax')(resnet_model.layers[-2].output)
    model = keras.Model(resnet_model.inputs, x)

    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_rot_train)

    history_SSL = model.fit(datagen.flow(x_rot_train, y_rot_train, batch_size=64), validation_data=(x_rot_test, y_rot_test), epochs=50, callbacks=callbacks, verbose=1)

    history_filepath = os.path.join(save_dir, model_name + "_history.pkl")
    with open(history_filepath, 'wb') as fp:
        pickle.dump(history_SSL.history, fp)

    print("Model and training history saved successfully.")


Train, Test = cifar10.load_data()
make_ssl_backbone(Train, Test, save_dir='saved_models')

print("Current Path:", os.getcwd())
import pandas as pd
with open("/content/saved_models/Restnetv1_SSL_Rotation.keras_history.pkl", 'rb') as fp:
    history = pickle.load(fp)

ssl_df = pd.DataFrame({'loss': history['loss'] + history['val_loss'], 'accuracy': history['accuracy'] + history['val_accuracy'], 'Dataset': 'train'})
ssl_df.loc[len(ssl_df)//2:, 'Dataset'] = 'validation'
ssl_df['epoch'] = np.concatenate((np.arange(len(history['loss'])), np.arange(len(history['loss']))))

import seaborn as sns
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(10, 5))
g = sns.lineplot(y='accuracy', x='epoch', hue='Dataset', data=ssl_df)
g.set_title('Accuracy - Self-Supervised Image Rotation Pretext Task')

figure = plt.figure(figsize=(10, 5))
g = sns.lineplot(y='loss', x='epoch', hue='Dataset', data=ssl_df)
g.set_title('Loss - Self-Supervised Image Rotation Pretext Task')
