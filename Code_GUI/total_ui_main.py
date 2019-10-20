#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from total_ui import Ui_MainWindow
from DCGAN import gan_g1, gan_g2, gan_g3, resnet_gen, gan_g5, dcgan_d1, dcgan_d2, resnet_dis, combined_g, train, train2, sample
from LSGAN import  train5, train6
from WGAN_div import wgan_d1, wgan_d2, resnet_dis_w,  combined_d_w, combined_g_w, train3, train4
from Classification import preprocess, plot_confusion_matrix, smooth_curve, plot_classification_report, train_cl
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
from keras.models import load_model
from keras import layers
from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Dropout, ZeroPadding2D, Reshape
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
import imageio
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support


#Ui_MainWindow, QtBaseClass = uic.loadUiType('classification_ui.ui')

class MyApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)
        self.show_images_button.clicked.connect(self.gan_visual)  ############ click to display
        self.dcgan_loops_button.clicked.connect(self.dcgan_loop_train)  ######### click to train
        self.dcgan_epochs_button.clicked.connect(self.dcgan_epoch_train)  ######### click to train
        self.wgan_loops_button.clicked.connect(self.wgan_loop_train)  ######### click to train
        self.wgan_epochs_button.clicked.connect(self.wgan_epoch_train)  ######### click to train
        self.lsgan_loops_button.clicked.connect(self.lsgan_loop_train)  ######### click to train
        self.lsgan_epochs_button.clicked.connect(self.lsgan_epoch_train)  ######### click to train
        self.train_npy_button.clicked.connect(self.select_raw)  ######### click to train
        self.select_model_button.clicked.connect(self.select_model)  ############ click to select model
        self.cl_epochs_button.clicked.connect(self.classification)  ############ click to classify
        self.select_loss_button.clicked.connect(self.cl_loss_visual)
        self.select_acc_button.clicked.connect(self.cl_acc_visual)
        self.select_cm_button.clicked.connect(self.cl_cm_visual)
        self.select_table_button.clicked.connect(self.cl_mt_visual)
        self.bs_gan_spinBox.setRange(0, 512)   #batch size
        self.bs_gan_spinBox.setSingleStep(1)
        self.epochs_gan_spinBox.setRange(0, 100000)  #loop/epoch
        self.epochs_gan_spinBox.setSingleStep(200)
        self.k_d_spinBox.setRange(0, 10)   #discriminator training time per loop/epoch
        self.k_d_spinBox.setSingleStep(1)
        self.k_g_spinBox.setRange(0, 10)  #generator training per loop/epoch
        self.k_g_spinBox.setSingleStep(1)
        self.interval_spinBox.setRange(0, 5000)   #saving interval
        self.interval_spinBox.setSingleStep(50)
        self.channel_spinBox.setRange(1,3)   #training image channel
        self.channel_spinBox.setSingleStep(2)
        self.z_dim_spinBox.setRange(0, 2000)  #latent vector size
        self.z_dim_spinBox.setSingleStep(20)
        self.display_spinBox.setRange(1, 10)   #display generate images combined size
        self.display_spinBox.setSingleStep(1)
        self.cl_pitting_amount_spinBox.setRange(0, 2000)  #generated pitting images amount
        self.cl_pitting_amount_spinBox.setSingleStep(50)
        self.optimizer_d_lr_spinBox.setRange(0, 1)  #adam optimizer learning rate of discriminator
        self.optimizer_d_lr_spinBox.setSingleStep(0.00005)
        self.optimizer_d_b_spinBox.setRange(0, 1) #adam optimizer beta 1 of discriminator
        self.optimizer_d_b_spinBox.setSingleStep(0.01)
        self.optimizer_g_lr_spinBox.setRange(0, 1)  #adam optimizer learning rate of generator
        self.optimizer_g_lr_spinBox.setSingleStep(0.00005)
        self.optimizer_g_b_spinBox.setRange(0, 1)  #adam optimizer beta 1 of generator
        self.optimizer_g_b_spinBox.setSingleStep(0.01)
        self.smooth_real_doubleSpinBox.setRange(0, 1)  #label smoothing in DCGAN for real images
        self.smooth_real_doubleSpinBox.setSingleStep(0.1)
        self.smooth_fake_doubleSpinBox.setRange(0, 1)  #label smoothing in DCGAN for GAN generated images
        self.smooth_fake_doubleSpinBox.setSingleStep(0.1)

        self.optimizer_cl_lr_spinBox.setRange(0, 1)  #classification learning rate
        self.optimizer_cl_lr_spinBox.setSingleStep(0.00005)
        self.optimizer_cl_b_spinBox.setRange(0, 1)  #classification adam optimizer beta 1
        self.optimizer_cl_b_spinBox.setSingleStep(0.01)
        self.epochs_cl_spinBox.setRange(0, 5000)  #training epoch
        self.epochs_cl_spinBox.setSingleStep(20)
        self.bs_cl_spinBox.setRange(0, 512)  #training batch size
        self.bs_cl_spinBox.setSingleStep(1)


    def select_raw(self):   #select trained original image dataset (np file)
        fname, _ = QFileDialog.getOpenFileName(self, "All Files (*)")
        file = os.path.basename(fname)
        self.gan_npy_box.setText(file)
        print(file)
        self.file_raw = file

    def gan_visual(self):  #select saved training image samples at different training epoch

        fname, _ = QFileDialog.getOpenFileName(self, "All Files (*)")
        self.gan_display_box.setScaledContents(True)
        self.gan_display_box.setPixmap(QPixmap(fname))
        
            
    def dcgan_loop_train(self):   #training DCGAN loop-wise

        save_file_gan_ex = self.gan_save_box.toPlainText()  #####input save folder for single experiments
        rounds = self.epochs_gan_spinBox.value()  #training loops
        bs = self.bs_gan_spinBox.value()  #batch size
        kd = self.k_d_spinBox.value()  #training D kd times per loop
        kg = self.k_g_spinBox.value()  #training G kg times per loop
        ndis = self.display_spinBox.value()  #row/column of generated display images
        n_interval = self.interval_spinBox.value()  #sacing interval
        channel_n = self.channel_spinBox.value()  #image channel
        z_dim = self.z_dim_spinBox.value()  #latent vector size
        in_r = self.smooth_real_doubleSpinBox.value()  ###label somooth for real images
        in_f = self.smooth_fake_doubleSpinBox.value()  ###label smooth for fake images
        save_file_gan = os.path.join('../dcgan_result/', str(save_file_gan_ex))  #saving path of the result
        lr_g = self.optimizer_g_lr_spinBox.value()
        b_g = self.optimizer_g_b_spinBox.value()
        lr_d = self.optimizer_d_lr_spinBox.value()
        b_d = self.optimizer_d_b_spinBox.value()
        real_dim = (96, 96, channel_n) #image size
        train_set = np.load(os.path.join('../npy/', str(self.file_raw)))  #path of trained original image dataset

        textg = self.gen_comboBox.currentText()  #select network
        if textg == 'gen1':
            gen = gan_g1(z_dim, channel_n)
        elif textg == 'gen2':
            gen = gan_g2(z_dim, channel_n)
        elif textg == 'gen3':
            gen = gan_g3(z_dim, channel_n)
        elif textg == 'gen4':
            gen = resnet_gen(z_dim, channel_n)
        elif textg == 'gen5':
            gen = gan_g5(z_dim, channel_n)

        textb = self.dis_comboBox.currentText()
        if textb == 'dis1':
            dis = dcgan_d1(real_dim)

        elif textb == 'dis2':
            dis = dcgan_d2(real_dim)

        elif textb == 'dis3':
            dis = resnet_dis(channel_n)

        optimizer_g = Adam(lr_g, b_g)
        optimizer_d = Adam(lr_d, b_d)

        if not os.path.exists('../dcgan_result'):  # create folder to save the experiment result.
            os.mkdir('../dcgan_result')
        if not os.path.exists(save_file_gan):
            os.mkdir(save_file_gan)

        train(X = train_set, G = gen, D = dis, loops = rounds, batch_size = bs, k_d = kd, k_g = kg,
              index_r = in_r, index_f = in_f, opt_g= optimizer_g, opt_d= optimizer_d, z_dim= z_dim,
              channel= channel_n, save_file = save_file_gan, n_i= n_interval, n = ndis)


    def dcgan_epoch_train(self):  #training DCGAN epoch-wise

        save_file_gan_ex = self.gan_save_box.toPlainText()  #####input save folder for single experiments
        rounds = self.epochs_gan_spinBox.value()
        bs = self.bs_gan_spinBox.value()
        kd = self.k_d_spinBox.value()
        kg = self.k_g_spinBox.value()
        n_interval = self.interval_spinBox.value()
        channel_n = self.channel_spinBox.value()
        z_dim = self.z_dim_spinBox.value()
        in_r = self.smooth_real_doubleSpinBox.value()  ###
        in_f = self.smooth_fake_doubleSpinBox.value()  ###
        save_file_gan = os.path.join('../dcgan_result/', str(save_file_gan_ex))
        lr_g = self.optimizer_g_lr_spinBox.value()
        b_g = self.optimizer_g_b_spinBox.value()
        lr_d = self.optimizer_d_lr_spinBox.value()
        b_d = self.optimizer_d_b_spinBox.value()
        real_dim = (96, 96, channel_n)
        train_set = np.load(os.path.join('../npy/', str(self.file_raw)))
        ndis = self.display_spinBox.value()

        textg = self.gen_comboBox.currentText()
        if textg == 'gen1':
            gen = gan_g1(z_dim, channel_n)
        elif textg == 'gen2':
            gen = gan_g2(z_dim, channel_n)
        elif textg == 'gen3':
            gen = gan_g3(z_dim, channel_n)
        elif textg == 'gen4':
            gen = resnet_gen(z_dim, channel_n)
        elif textg == 'gen5':
            gen = gan_g5(z_dim, channel_n)

        textb = self.dis_comboBox.currentText()
        if textb == 'dis1':
            dis = dcgan_d1(real_dim)

        elif textb == 'dis2':
            dis = dcgan_d2(real_dim)

        elif textb == 'dis3':
            dis = resnet_dis(channel_n)

        optimizer_g = Adam(lr_g, b_g)
        optimizer_d = Adam(lr_d, b_d)

        if not os.path.exists('../dcgan_result'):  # create folder to save the experiment result.
            os.mkdir('../dcgan_result')
        if not os.path.exists(save_file_gan):
            os.mkdir(save_file_gan)

        train2(X=train_set, G=gen, D=dis, nb_epochs=rounds, batch_size=bs, k_d=kd, k_g=kg,
              index_r=in_r, index_f=in_f, opt_g=optimizer_g, opt_d=optimizer_d, z_dim=z_dim,
              channel=channel_n, save_file=save_file_gan, n_i=n_interval, n = ndis)

    def wgan_loop_train(self):  #training WGAN-div loop-wise
        save_file_gan_ex = self.gan_save_box.toPlainText()  #####input save folder for single experiments
        rounds = self.epochs_gan_spinBox.value()
        bs = self.bs_gan_spinBox.value()
        kd = self.k_d_spinBox.value()
        kg = self.k_g_spinBox.value()
        ndis = self.display_spinBox.value()
        n_interval = self.interval_spinBox.value()
        channel_n = self.channel_spinBox.value()
        z_dim = self.z_dim_spinBox.value()
        save_file_gan = os.path.join('../wgan_result/', str(save_file_gan_ex))
        lr_g = self.optimizer_g_lr_spinBox.value()
        b_g = self.optimizer_g_b_spinBox.value()
        lr_d = self.optimizer_d_lr_spinBox.value()
        b_d = self.optimizer_d_b_spinBox.value()
        real_dim = (96, 96, channel_n)
        train_set = np.load(os.path.join('../npy/', str(self.file_raw)))

        textg = self.gen_comboBox.currentText()
        if textg == 'gen1':
            gen = gan_g1(z_dim, channel_n)
        elif textg == 'gen2':
            gen = gan_g2(z_dim, channel_n)
        elif textg == 'gen3':
            gen = gan_g3(z_dim, channel_n)
        elif textg == 'gen4':
            gen = resnet_gen(z_dim, channel_n)
        elif textg == 'gen5':
            gen = gan_g5(z_dim, channel_n)

        textb = self.dis_comboBox.currentText()
        if textb == 'dis4':
            dis = wgan_d1(real_dim)

        elif textb == 'dis5':
            dis = wgan_d2(real_dim)

        elif textb == 'dis6':
            dis = resnet_dis_w(channel_n)

        optimizer_g = Adam(lr_g, b_g)
        optimizer_d = Adam(lr_d, b_d)


        if not os.path.exists('../wgan_result'):  # create folder to save the experiment result.
            os.mkdir('../wgan_result')
        if not os.path.exists(save_file_gan):
            os.mkdir(save_file_gan)

        train3(X=train_set, G=gen, D=dis, loops=rounds, batch_size=bs, k_d=kd, k_g=kg,
              opt_g=optimizer_g, opt_d=optimizer_d, z_dim=z_dim,
              channel=channel_n, save_file=save_file_gan, n_i=n_interval, n=ndis, real_dim = real_dim)

    def wgan_epoch_train(self):  #training wgan-div epoch-wise
        save_file_gan_ex = self.gan_save_box.toPlainText()  #####input save folder for single experiments
        rounds = self.epochs_gan_spinBox.value()
        bs = self.bs_gan_spinBox.value()
        kd = self.k_d_spinBox.value()
        kg = self.k_g_spinBox.value()
        ndis = self.display_spinBox.value()
        n_interval = self.interval_spinBox.value()
        channel_n = self.channel_spinBox.value()
        z_dim = self.z_dim_spinBox.value()
        save_file_gan = os.path.join('../wgan_result/', str(save_file_gan_ex))
        lr_g = self.optimizer_g_lr_spinBox.value()
        b_g = self.optimizer_g_b_spinBox.value()
        lr_d = self.optimizer_d_lr_spinBox.value()
        b_d = self.optimizer_d_b_spinBox.value()
        real_dim = (96, 96, channel_n)
        train_set = np.load(os.path.join('../npy/', str(self.file_raw)))

        textg = self.gen_comboBox.currentText()
        if textg == 'gen1':
            gen = gan_g1(z_dim, channel_n)
        elif textg == 'gen2':
            gen = gan_g2(z_dim, channel_n)
        elif textg == 'gen3':
            gen = gan_g3(z_dim, channel_n)
        elif textg == 'gen4':
            gen = resnet_gen(z_dim, channel_n)
        elif textg == 'gen5':
            gen = gan_g5(z_dim, channel_n)

        textb = self.dis_comboBox.currentText()
        if textb == 'dis4':
            dis = wgan_d1(real_dim)

        elif textb == 'dis5':
            dis = wgan_d2(real_dim)

        elif textb == 'dis6':
            dis = resnet_dis_w(channel_n)

        optimizer_g = Adam(lr_g, b_g)
        optimizer_d = Adam(lr_d, b_d)


        if not os.path.exists('../wgan_result'):  # create folder to save the experiment result.
            os.mkdir('../wgan_result')
        if not os.path.exists(save_file_gan):
            os.mkdir(save_file_gan)

        train4(X=train_set, G=gen, D=dis, nb_epochs=rounds, batch_size=bs, k_d=kd, k_g=kg,
              opt_g=optimizer_g, opt_d=optimizer_d, z_dim=z_dim,
              channel=channel_n, save_file=save_file_gan, n_i=n_interval, n=ndis, real_dim = real_dim)


    def lsgan_loop_train(self):  #training LSGAN loop-wise

        save_file_gan_ex = self.gan_save_box.toPlainText()  #####input save folder for single experiments
        rounds = self.epochs_gan_spinBox.value()
        bs = self.bs_gan_spinBox.value()
        kd = self.k_d_spinBox.value()
        kg = self.k_g_spinBox.value()
        ndis = self.display_spinBox.value()
        n_interval = self.interval_spinBox.value()
        channel_n = self.channel_spinBox.value()
        z_dim = self.z_dim_spinBox.value()
        in_r = self.smooth_real_doubleSpinBox.value()  ###
        in_f = self.smooth_fake_doubleSpinBox.value()  ###
        save_file_gan = os.path.join('../lsgan_result/', str(save_file_gan_ex))
        lr_g = self.optimizer_g_lr_spinBox.value()
        b_g = self.optimizer_g_b_spinBox.value()
        lr_d = self.optimizer_d_lr_spinBox.value()
        b_d = self.optimizer_d_b_spinBox.value()
        real_dim = (96, 96, channel_n)
        train_set = np.load(os.path.join('../npy/', str(self.file_raw)))

        textg = self.gen_comboBox.currentText()
        if textg == 'gen1':
            gen = gan_g1(z_dim, channel_n)
        elif textg == 'gen2':
            gen = gan_g2(z_dim, channel_n)
        elif textg == 'gen3':
            gen = gan_g3(z_dim, channel_n)
        elif textg == 'gen4':
            gen = resnet_gen(z_dim, channel_n)
        elif textg == 'gen5':
            gen = gan_g5(z_dim, channel_n)

        textb = self.dis_comboBox.currentText()    #discriminator of LSGAN are the same as that in WGAN-div, all belongs to solve the regression task,which does not have sigmoid activation function in the output layer
        if textb == 'dis4':
            dis = wgan_d1(real_dim)

        elif textb == 'dis5':
            dis = wgan_d2(real_dim)

        elif textb == 'dis6':
            dis = resnet_dis_w(channel_n)

        optimizer_g = Adam(lr_g, b_g)
        optimizer_d = Adam(lr_d, b_d)

        if not os.path.exists('../lsgan_result'):  # create folder to save the experiment result.
            os.mkdir('../lsgan_result')
        if not os.path.exists(save_file_gan):
            os.mkdir(save_file_gan)

        train5(X=train_set, G=gen, D=dis, loops=rounds, batch_size=bs, k_d=kd, k_g=kg,
              index_r=in_r, index_f=in_f, opt_g=optimizer_g, opt_d=optimizer_d, z_dim=z_dim,
              channel=channel_n, save_file=save_file_gan, n_i=n_interval, n=ndis)

    def lsgan_epoch_train(self):  #training LSGAN epoch-wise

        save_file_gan_ex = self.gan_save_box.toPlainText()  #####input save folder for single experiments
        rounds = self.epochs_gan_spinBox.value()
        bs = self.bs_gan_spinBox.value()
        kd = self.k_d_spinBox.value()
        kg = self.k_g_spinBox.value()
        n_interval = self.interval_spinBox.value()
        channel_n = self.channel_spinBox.value()
        z_dim = self.z_dim_spinBox.value()
        in_r = self.smooth_real_doubleSpinBox.value()  ###
        in_f = self.smooth_fake_doubleSpinBox.value()  ###
        save_file_gan = os.path.join('../lsgan_result/', str(save_file_gan_ex))
        lr_g = self.optimizer_g_lr_spinBox.value()
        b_g = self.optimizer_g_b_spinBox.value()
        lr_d = self.optimizer_d_lr_spinBox.value()
        b_d = self.optimizer_d_b_spinBox.value()
        real_dim = (96, 96, channel_n)
        train_set = np.load(os.path.join('../npy/', str(self.file_raw)))
        ndis = self.display_spinBox.value()

        textg = self.gen_comboBox.currentText()
        if textg == 'gen1':
            gen = gan_g1(z_dim, channel_n)
        elif textg == 'gen2':
            gen = gan_g2(z_dim, channel_n)
        elif textg == 'gen3':
            gen = gan_g3(z_dim, channel_n)
        elif textg == 'gen4':
            gen = resnet_gen(z_dim, channel_n)
        elif textg == 'gen5':
            gen = gan_g5(z_dim, channel_n)

        textb = self.dis_comboBox.currentText()
        if textb == 'dis4':
            dis = wgan_d1(real_dim)

        elif textb == 'dis5':
            dis = wgan_d2(real_dim)

        elif textb == 'dis6':
            dis = resnet_dis_w(channel_n)

        optimizer_g = Adam(lr_g, b_g)
        optimizer_d = Adam(lr_d, b_d)

        if not os.path.exists('../lsgan_result'):  # create folder to save the experiment result.
            os.mkdir('../lsgan_result')
        if not os.path.exists(save_file_gan):
            os.mkdir(save_file_gan)

        train6(X=train_set, G=gen, D=dis, nb_epochs=rounds, batch_size=bs, k_d=kd, k_g=kg,
               index_r=in_r, index_f=in_f, opt_g=optimizer_g, opt_d=optimizer_d, z_dim=z_dim,
               channel=channel_n, save_file=save_file_gan, n_i=n_interval, n=ndis)

########################################################################################################################
#####below is for classification task

    def select_model(self):
        fname, _ = QFileDialog.getOpenFileName(self,"All Files (*)")
        file = fname.split("/")[-3] + "/"+fname.split("/")[-2] + "/" + fname.split("/")[-1]  #extract model name
        self.GAN_model_box.setText(file)
        print(file)
        self.model = file

    def classification(self):

        save_cl_gan_ex = self.result_saving_cl_box.toPlainText()  #####input save folder for single experiments
        save_cl_gan = os.path.join('../classification_result/', str(save_cl_gan_ex))
        z_dim = self.z_dim_spinBox.value()
        model_cl = load_model(os.path.join('../', str(self.model)))
        n = self.cl_pitting_amount_spinBox.value()   #generated GAN images, recommend 700 to be the same as the number of train+valid set of intsct images
        z_sample = np.random.randn(n, z_dim)
        x_sample = model_cl.predict(z_sample)
        x_sample = 127.5 * x_sample + 127.5
        x_sample = np.round(x_sample, 0).astype(np.uint8)
        bscl = self.bs_cl_spinBox.value()  #classification batch size


        x_sample_gray = []
        if x_sample.shape[-1]==3:
            for ig in range(len(x_sample)):
                imgs_raw = cv2.cvtColor(x_sample[ig], cv2.COLOR_RGB2GRAY)  # convert gray images
                x_sample_gray.append(imgs_raw)
            x_sample_gray = np.array(x_sample_gray)

        if x_sample.shape[-1]!=3:
            x_sample_gray = x_sample
            x_sample_gray = np.squeeze(x_sample_gray, axis=-1)

        i_train = np.load('../npy/intact_train.npy')  #
        i_valid = np.load('../npy/intact_valid.npy')
        i_test = np.load('../npy/intact_test.npy')
        f_test = np.load('../npy/pitting_test.npy')

        # labelize, 0 for flaking, 1 for intact
        label_fe = np.zeros(len(f_test)).astype(np.int)
        label_it = np.ones(len(i_train)).astype(np.int)
        label_iv = np.ones(len(i_valid)).astype(np.int)
        label_ie = np.ones(len(i_test)).astype(np.int)
        pitting_label = np.zeros(len(x_sample)).astype(np.int)

        f_train, f_valid, label_ft, label_fv = train_test_split(x_sample_gray, pitting_label,
                                                                test_size=0.3, random_state=0)
        print(f_train.shape, i_train.shape)

        total_train = np.concatenate((f_train, i_train), axis=0)
        total_train_l = np.concatenate((label_ft, label_it), axis=0)
        total_valid = np.concatenate((f_valid, i_valid), axis=0)
        total_valid_l = np.concatenate((label_fv, label_iv), axis=0)
        total_test = np.concatenate((f_test, i_test), axis=0)
        total_test_l = np.concatenate((label_fe, label_ie), axis=0)

        x_train, y_train = preprocess(total_train, total_train_l)  #stack, resize and label one-hot encoding
        x_valid, y_valid = preprocess(total_valid, total_valid_l)
        x_test, y_test = preprocess(total_test, total_test_l)

        # shuffle the data
        x_train_s, y_train_s = shuffle(x_train, y_train, random_state=25)
        x_valid_s, y_valid_s = shuffle(x_valid, y_valid, random_state=25)
        x_test_s, y_test_s = shuffle(x_test, y_test, random_state=25)

        if not os.path.exists('../classification_result'):  # create folder to save the experiment result.
            os.mkdir('../classification_result')
        if not os.path.exists(save_cl_gan):
            os.mkdir(save_cl_gan)

        train_cl(x_train = x_train_s, y_train = y_train_s, x_valid = x_valid_s, y_valid = y_valid_s, x_test = x_test_s,
                 y_test = y_test_s, epochs = self.epochs_cl_spinBox.value(), save= save_cl_gan + '/' + save_cl_gan_ex + '.pckl' ,
                 adam_lr = self.optimizer_cl_lr_spinBox.value(), adam_b = self.optimizer_cl_b_spinBox.value(), save_folder = save_cl_gan, bs_cl=bscl)

    def cl_loss_visual(self):  #loss curve
        fname, _ = QFileDialog.getOpenFileName(self,  "All Files (*)")
        self.loss_image_box.setScaledContents(True)
        self.loss_image_box.setPixmap(QPixmap(fname))

    def cl_acc_visual(self):  #accuracy curve
        fname, _ = QFileDialog.getOpenFileName(self,  "All Files (*)")
        self.acc_image_box.setScaledContents(True)
        self.acc_image_box.setPixmap(QPixmap(fname))

    def cl_cm_visual(self):   #confusion matrix
        fname, _ = QFileDialog.getOpenFileName(self, "All Files (*)")
        self.cm_box.setScaledContents(True)
        self.cm_box.setPixmap(QPixmap(fname))

    def cl_mt_visual(self):  #evaluation metrics
        fname, _ = QFileDialog.getOpenFileName(self,  "All Files (*)")
        self.metrics_box.setScaledContents(True)
        self.metrics_box.setPixmap(QPixmap(fname))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())








