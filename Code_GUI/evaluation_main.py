#!/usr/bin/env python
# coding: utf-8

# In[1]:


from evaluation_ui import Ui_MainWindow
from FID import fid_calc
from tsne import tsne_visual
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5.QtGui import *
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
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.models import load_model
#import matplotlib.pyplot as plt
from qimage2ndarray import array2qimage
import time
import functools
from six.moves import xrange
import imageio
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#Ui_MainWindow, QtBaseClass = uic.loadUiType('classification_ui.ui')


class MyApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)
        self.model_button.clicked.connect(self.gan_model)
        self.original_button.clicked.connect(self.original_load)
        self.random_button.clicked.connect(self.sample)
        self.spinBox.setRange(0, 512) #row/column- of display image
        self.spinBox.setSingleStep(1)
        self.z_dim_spinBox.setRange(0, 2000) #latent vector size
        self.z_dim_spinBox.setSingleStep(20)
        self.spinBox_tsne.setRange(0, 2000) #number of generated images for tsne display
        self.spinBox_tsne.setSingleStep(50)
        self.fid_button.clicked.connect(self.generate)
        self.tsne_button.clicked.connect(self.tsne_plot)

    def gan_model(self): #choose trained generator model
        fname, _ = QFileDialog.getOpenFileName(self,"All Files (*)")
        file = fname.split("/")[-3] + "/"+fname.split("/")[-2] + "/" + fname.split("/")[-1]
        self.model_box.setText(file)
        print(file)
        self.model = file

    def original_load(self): #choose trained generator model
        fname, _ = QFileDialog.getOpenFileName(self,"All Files (*)")
        file = fname.split("/")[-2] + "/"+fname.split("/")[-1]
        self.original_box.setText(file)
        print(file)
        self.real = file

    def sample(self):  #display generated images
        modelsa = load_model(os.path.join('../', str(self.model)))
        n = self.spinBox.value()
        img_dim = 96
        figure = np.zeros((img_dim * n, img_dim * n, 3))
        for i in range(n):
            for j in range(n):
                z_sample = np.random.randn(1, self.z_dim_spinBox.value())
                x_sample = modelsa.predict(z_sample)  # create one generated image
                digit = x_sample[0]
                figure[i * img_dim:(i + 1) * img_dim,
                j * img_dim:(j + 1) * img_dim] = digit  # put a generated image into the corresponding position.
        figure = 127.5 * figure + 127.5  # convert generated images from [-1,1] to [0,255]
        figure = np.round(figure, 0).astype(np.uint8)  # make sure the image value is type uint8 (integer from 0 to 255)
        figure_im = array2qimage(figure)
        self.display_box.setScaledContents(True)
        self.display_box.setPixmap(QPixmap(figure_im))

    def generate(self):
        real = np.load(os.path.join('../', str(self.real)))
        real = (real-127.5)/127.5
        modelge = load_model(os.path.join('../', str(self.model)))
        z_sample = np.random.randn(len(real), self.z_dim_spinBox.value())
        x_sample = modelge.predict(z_sample)
        #x_sample = 127.5 * x_sample + 127.5
        #x_sample = np.round(x_sample, 0).astype(np.uint8)
        fid_value = fid_calc(x_sample, real)
        self.lineEdit.setText('{}'.format(fid_value))
        print(fid_value)

    def tsne_plot(self):
        real = np.load(os.path.join('../', str(self.original)))
        real = (real - 127.5) / 127.5
        modelts = load_model(os.path.join('../', str(self.model)))
        n = self.spinBox_tsne.value()
        z_sample = np.random.randn(n, self.z_dim_spinBox.value())
        x_sample = modelts.predict(z_sample)
        tsne_visual(x_sample, real)
        img = plt.imread("tsne.jpg")
        img = np.round(img, 0).astype(np.uint8)  # make sure the image value is type uint8 (integer from 0 to 255)
        img_im = array2qimage(img)
        self.tsne_box.setScaledContents(True)
        self.tsne_box.setPixmap(QPixmap(img_im))
        os.remove("tsne.jpg")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    




