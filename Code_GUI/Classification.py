#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# code for section 4.5.3.1 Classification only with GAN images 

import numpy as np
import tensorflow as tf
import random as rn
import os
import keras
import cv2
import glob
import pickle
import os
from keras import models
from keras import layers
from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Dropout, ZeroPadding2D
from keras.utils import np_utils
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support


def preprocess(x_input, y_input):    
    x_group = []

    #stack gray images to 3 channels
    for k in range(len(x_input)):
        resized = x_input[k]
        resized_3 = np.zeros((resized.shape[0], resized.shape[1], 3), "uint8")
        resized_3[:, :, 0], resized_3[:, :, 1], resized_3[:, :, 2] = resized, resized, resized
        x_group.append(resized_3)
    x_group = np.array(x_group)
    #normalise the input dataset 
    x_group = (x_group-127.5)/127.5
    #transfer the label to one-hot-encoding
    y = to_categorical(y_input)
    return x_group, y

def plot_confusion_matrix(confusion_mat, save_folder):

    import matplotlib.pyplot as plt

    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix   0-pitting 1-intact')
    plt.colorbar()
    m=np.arange(2)
    plt.xticks(m)
    plt.yticks(m)
    for s1 in range(len(m)):
        for s2 in range(len(m)):
            plt.text(s1,s2,confusion_mat[s2][s1],fontsize=15,color='white')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_folder + '/confusion_matrix.jpg', bbox_inches='tight')
    plt.close('all')

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else: smoothed_points.append(point)
    return smoothed_points


def plot_classification_report(y_tru, y_prd, save_folder):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    xticks = ['precision', 'recall', 'f1-score', 'amount']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True,
                cbar=False,
                xticklabels=xticks,
                yticklabels=yticks,
                cmap=ListedColormap(['white'])
                )

    plt.savefig(save_folder + '/report.jpg', bbox_inches='tight')
    plt.close('all')


def train_cl(x_train, y_train, x_valid, y_valid, x_test, y_test, epochs, save, adam_lr, adam_b, save_folder, bs_cl):

    from keras.applications.inception_v3 import InceptionV3
    import matplotlib.pyplot as plt

    freeze_flag = True  # `True` to freeze layers, `False` for full training
    weights_flag = 'imagenet' # 'imagenet' or None
    preprocess_flag = True # Should be true for ImageNet pre-trained typically

    input_size1 = 96
    input_size2 = 96

    # Using Inception with ImageNet pre-trained weights
    inception = InceptionV3(weights=weights_flag, include_top=False,
                          input_shape=(input_size2,input_size1,3))

    if freeze_flag == True:
        for layer in inception.layers:
            layer.trainable = False

    from keras.layers import GlobalAveragePooling2D
    from keras.layers import Input

    img_input = Input(shape=(input_size2,input_size1,3))
    inp = inception(img_input)

    x = GlobalAveragePooling2D()(inp)
    x = Dense(128,kernel_regularizer=regularizers.l2(0.002),activation = 'relu')(x)
    x = Dense(16,activation = 'relu')(x)
    predictions = Dense(2, activation = 'softmax')(x)

    from keras.models import Model

    model = Model(inputs=img_input, outputs=predictions)
    optimizer = Adam(adam_lr, adam_b)
    #optimizer = RMSprop(lr=0.001)

    #stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    model.compile(optimizer = optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=bs_cl, validation_data= (x_valid, y_valid), verbose=0)



    f = open(save, 'wb')
    pickle.dump(history.history, f)
    f.close()


    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    test_prediction = model.predict(x_test)

    print('loss: ',test_loss, ' acc: ', test_acc)

    label_recover = np.argmax(test_prediction, axis=1)
    label_ori_recover = np.argmax(y_test, axis=1)
    confusion_mat = confusion_matrix(label_ori_recover, label_recover)
    plot_confusion_matrix(confusion_mat, save_folder)

    plt.plot(epochs, smooth_curve(acc), 'r', label='Smoothed training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(save_folder + '/acc_tv.jpg', bbox_inches='tight')
    plt.close('all')


    plt.plot(epochs, smooth_curve(loss_values), 'r', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss_values), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(save_folder + '/loss_tv.jpg', bbox_inches='tight')
    plt.close('all')

    plot_classification_report(label_ori_recover, label_recover, save_folder)

    x_test_s = cv2.normalize(x_test, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    wrong_images = []
    wrong_labels = []
    for w in range(len(x_test_s)):
        if label_ori_recover[w] != label_recover[w]:
            wrong_images.append(x_test_s[w])
            wrong_labels.append(w)

    plt.figure(figsize=(50, 50))

    for g in range(len(wrong_images)):

        plt.subplot(16, 16, 1 + g)
        plt.imshow(wrong_images[g], cmap='gray')
        plt.text(2, 10, 'True_Label:', fontsize=10, color='red')
        plt.text(2, 20, label_ori_recover[wrong_labels[g]], fontsize=10, color='red')
        plt.text(50, 10, 'Pred_Label:', fontsize=10, color='red')
        plt.text(50, 20, label_recover[wrong_labels[g]], fontsize=10, color='red')
        plt.axis('off')
    plt.savefig(save_folder + '/fault_cl1.jpg', bbox_inches='tight')
    plt.close('all')

    plt.figure(figsize=(50, 50))
    for aa in range(len(x_test_s)):

        plt.subplot(16, 16, 1 + aa)
        plt.imshow(x_test_s[aa], cmap='gray')
        if label_ori_recover[aa] == label_recover[aa]:
            plt.text(2, 10, 'True_Label:', fontsize=10, color='green')
            plt.text(2, 20, label_ori_recover[aa], fontsize=10, color='green')
            plt.text(50, 10, 'Pred_Label:', fontsize=10, color='green')
            plt.text(50, 20, label_recover[aa], fontsize=10, color='green')
        if label_ori_recover[aa] != label_recover[aa]:
            plt.text(2, 10, 'True_Label:', fontsize=10, color='red')
            plt.text(2, 20, label_ori_recover[aa], fontsize=10, color='red')
            plt.text(50, 10, 'Pred_Label:', fontsize=10, color='red')
            plt.text(50, 20, label_recover[aa], fontsize=10, color='red')

        plt.axis('off')
    plt.savefig(save_folder + '/fault_cl2.jpg', bbox_inches='tight')
    plt.close('all')




