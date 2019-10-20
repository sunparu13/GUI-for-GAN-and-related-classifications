#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.layers import *
from keras import backend as K
import cv2
import glob
import imageio
import sys
import os
import pickle
import cv2
import numpy as np

#total_train = np.concatenate((p_train, p_cda), axis=0)
# generator structures are the same as being used in DCGAN.py

def wgan_d1(input_shape_d):

    model = Sequential()
    
    model.add(GaussianNoise(stddev=0.02))   #add some noise to the first layer, with std=0.02
    model.add(Conv2D(32, kernel_size=5, strides=2, input_shape= input_shape_d, padding="same")) 
    #suppose input size is  96*96*1, than output size is: 48*48*32
    #kernel_initializer= RandomNormal(mean=0.0, stddev=0.2) can be added be selected
    model.add(LeakyReLU(alpha=0.2)) #leaky ReLu activation function with the gradient 0.2 for the negative part.
    model.add(Dropout(0.25)) #drop 25% neurons randomly
    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same")) # output size is 24*24*64, with 64 kernel filter, each one with size 5*5*32
    model.add(BatchNormalization()) #BN layer
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) # output size is 12*12*128, with 128 kernel filter, each one with size 3*3*64
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) # without upscaling because stride is 1.  output size is 12*12*256, with 256 kernel filter, each one with size 3*3*128
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten()) #flatten layer as the input of FCN layer, neuron number is 12*12*256=36864
    model.add(Dense(1)) #output without any activation function, it does an regression task.
    input_d = Input(shape=input_shape_d)
    output_d = model(input_d)
    return Model(input_d, output_d)

def wgan_d2(input_shape_d):

    model = Sequential()
    
    model.add(GaussianNoise(stddev=0.02))  #add some noise to the first layer
    model.add(Conv2D(32, kernel_size=4, strides=2, input_shape= input_shape_d, padding="same")) 
    # suppose input shape is 96*96*1, output size is 48*48*32
    #kernel_initializer= RandomNormal(mean=0.0, stddev=0.2) can be added be selected
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same")) #output is 24*24*64
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same")) #output is 12*12*256
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same")) #output is 6*6*512
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Dropout(0.25))
    model.add(GlobalAveragePooling2D())  #for each layer, calculate the average value and extract it, so output number is 512
    model.add(Dense(1)) #output without any activation function, it does an regression task..
    input_d = Input(shape=input_shape_d)
    output_d = model(input_d)
    return Model(input_d, output_d)


def resnet_dis_w(channel):
    
    """Resnet Discriminator"""
    def residual_block(layer_input, df):
        """Residual block described in paper"""
        d = BatchNormalization(momentum=0.8)(layer_input)
        d = Activation('relu')(d)
        d = Conv2D(df, kernel_size=4, strides=1, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = Activation('relu')(d)
        d = Conv2D(df, kernel_size=4, strides=1, padding='same')(d)
        d = Add()([d, layer_input])
        return d
    
    x_in = Input(shape=(96, 96, channel))
    x = x_in
    
    l1 = Conv2D(32, kernel_size=4, padding='same', activation='relu')(x)
    # Propogate signal through residual blocks
    r = residual_block(l1,32)
    #repear pooling and res block
    for _ in range(3):
        r = AveragePooling2D()(r)
        r = residual_block(r, 32)
     
    o = BatchNormalization(momentum=0.8)(r)
    o = Activation('relu')(o)
    o = Flatten()(o) #flatten layer as the input of FCN layer, neuron number is 12*12*256=36864
    o = Dense(1, activation='sigmoid')(o) #output is only one to s
    
    dis = Model(x_in, o)
    dis.summary()
    return dis

# combined Discriminator and combined entire GAN with
# fixed Generator used for training
#please see the original paper https://arxiv.org/pdf/1712.01026.pdf for more detail of the loss function of D and G

#combined discriminator with frozen GENERATOR
def combined_d_w(k, p, G, D, opt_d, z_dim, real_dim, bs):
    # according to the original wasserstein divergence paper, 
    # k=2 and p=6 has the best genereated effect
    from keras.layers.merge import _Merge
    class RandomWeightedAverage(_Merge):
        # Provides a (random) weighted average between real and generated image samples"""
        def _merge_function(self, inputs):
            alpha = K.random_uniform((bs, 1, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


    frozen_G = Model(
            inputs=G.inputs,
            outputs=G.outputs)
    frozen_G.trainable = False   #generator is frozen

    x_in = Input(shape=(real_dim))
    z_in = Input(shape=(z_dim, ))
    x_real = x_in
    x_fake = frozen_G(z_in)

    # output for the seperate real image batch and fake image batch
    x_real_score = D(x_real)
    x_fake_score = D(x_fake)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([x_real, x_fake])
    # Determine validity of weighted sample
    interpolated_score = D(interpolated_img)

    #output of training discriminator is 3 values, judge the real, fake and synthetic images
    d_train_model = Model([x_in, z_in],
                          [x_real_score, x_fake_score, interpolated_score])


    d_loss_init = K.mean(x_real_score - x_fake_score)

    # compute the gradient of the discriminator outrput with respect to input image
    # K is keras backend
    real_grad = K.gradients(x_real_score, [x_real])[0]
    fake_grad = K.gradients(x_fake_score, [x_fake])[0]
    interpolate_grad = K.gradients(x_fake_score, [x_fake])[0]

    # calculate the L2 norm to the power of p (‖∇T‖^p) sepereately for real batch and fake batch
    intl_grad_norm = K.sum(interpolate_grad**2, axis=[1, 2, 3])**(p / 2)
    real_grad_norm = K.sum(real_grad ** 2, axis=[1, 2, 3]) ** (p / 2)
    fake_grad_norm = K.sum(fake_grad ** 2, axis=[1, 2, 3]) ** (p / 2)
    grad_loss = K.mean(intl_grad_norm+real_grad_norm+fake_grad_norm) * k/3

    # d_loss to be minimized (maximize objective D)
    w_dist = K.mean(x_fake_score - x_real_score)
    d_train_model.add_loss(d_loss_init + grad_loss)
    #d_train_model.metrics_names.append('w_dist')
    #d_train_model.metrics_tensors.append(w_dist)

    #compile discriminator with Adam optimizer, the metric is the online calculated wasserstein distance
    d_train_model.compile(optimizer = opt_d)

    return d_train_model


# train G and freeze D
def combined_g_w(G, D, opt_g, z_dim):
    frozen_D = Model(
            inputs=D.inputs,
            outputs=D.outputs)
    frozen_D.trainable = False
    z_in = Input(shape=(z_dim,))
    x_fake = G(z_in)
    x_fake_prob = frozen_D(x_fake)

    gan_train_model = Model(z_in, x_fake_prob)

    # g_loss is to minimize the wasserstein distance 
    #w_dist = K.mean(x_fake_prob - x_real_prob), x_real_prob
    # is set to 0 beacause or the irrelevant to the generator
    g_loss = K.mean(x_fake_prob)
    gan_train_model.add_loss(g_loss)
    gan_train_model.compile(optimizer= opt_g)
    
    return gan_train_model


#form 10x10 connected images, each of them is 96x96
def sample(path,G,channel, z_dim, n):
    img_dim = 96
    figure = np.zeros((img_dim * n, img_dim * n, channel))
    for i in range(n):
        for j in range(n):
            z_sample = np.random.randn(1, z_dim)   
            x_sample = G.predict(z_sample)     #create one genrated image
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit  #put a generated image into the corresponding position.
    figure = 127.5*figure + 127.5   #convert generated images from [-1,1] to [0,255]
    figure = np.round(figure, 0).astype(np.uint8)  #make sure the image vaue is type uint8 (integer from 0 to 255)
    imageio.imwrite(path, figure)
    

def train3(X, G, D, loops, batch_size, k_d, k_g, opt_g, opt_d, z_dim, channel, save_file, n_i, n, real_dim):

    import matplotlib.pyplot as plt

    D_train = combined_d_w(2, 6, G, D, opt_d, z_dim, real_dim, batch_size)
    G_train = combined_g_w(G, D, opt_g, z_dim)
    
    lossesd = []
    lossesg = []
    
    # Rescale image from [0,255] to [-1 to 1]
    X = (X - 127.5)/127.5
    if X.shape[-1]!=3:  
        X = np.expand_dims(X, axis=3)  #if gray images, than add a dimension with the channel nummber 1.

    # in original WGAN-div paper, D and G are equally trained in one loop
    for loop in range(loops+1):
        for _ in range(k_d): #range can be tuned for further test
            
            idx = np.random.randint(0, X.shape[0], batch_size)  #random select batch size real images 
            imgs = X[idx]
            
            # Sample noise and generate a batch of new images, in further spherical sample with 'slerp' can be tried
            z_sample = np.random.normal(0, 1, (batch_size, z_dim)) #normal distribution selection of latent variable with the number z_dim for the batch
            d_loss = D_train.train_on_batch([imgs, z_sample], None)  #without label, it is a regression task to approximate the wasserstein distance
        for _ in range(k_g): #range can be tuned for further test
            z_sample = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = G_train.train_on_batch(z_sample, None)
            
        lossesd.append(d_loss)
        lossesg.append(g_loss)

        if loop % n_i == 0:
            print ("%d [D loss: %s] [G loss: %s]" % (loop, d_loss , g_loss))
            G.save(save_file + '/generator_model_%s_z_%s.h5' % (loop, z_dim))
            G_train.save(save_file + '/generator_train_model_%s_z_%s.h5' % (loop, z_dim))
            D_train.save(save_file + '/discriminator_train_model_%s_z_%s.h5' % (loop, z_dim))
            path = save_file + '/display_%s.png'
            sample(path % loop, G, channel, z_dim, n)
        
    fig, ax = plt.subplots()
    lossesd = np.array(lossesd)
    lossesg = np.array(lossesg)
    plt.plot(lossesd, label='Discriminator')
    plt.plot(lossesg, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(save_file + '/loop_loss.jpg', bbox_inches='tight')
    plt.close('all')


from sklearn.utils import shuffle

def train4(X, G, D, nb_epochs, batch_size, k_d, k_g, opt_g, opt_d, z_dim, channel, save_file, n_i, n, real_dim):

    import matplotlib.pyplot as plt

    D_train = combined_d_w(2, 6, G, D, opt_d, z_dim, real_dim, batch_size)
    G_train = combined_g_w(G, D, opt_g, z_dim)

    X=shuffle(X)
    # Rescale image from [0,255] to [-1 to 1]
    X = (X - 127.5)/127.5
    if X.shape[-1]!=3:  
        X = np.expand_dims(X, axis=3)  #if gray images, than add a dimension with the channel nummber 1.

    for epoch in range(nb_epochs+1):

        nb_batches = int(X.shape[0] / batch_size)

        epoch_gen_loss = []
        epoch_disc_loss_s = []
        epoch_disc_loss = []
        
        for _ in range(k_d):

            for index in range(nb_batches):

                # generate a new batch of noise
                z_sample = np.random.normal(0, 1, (batch_size, z_dim)) 

                # get a batch of real images
                image_batch = X[index * batch_size:(index + 1) * batch_size]  #go through batch to batch

                d_loss = D_train.train_on_batch([image_batch, z_sample], None)
                epoch_disc_loss_s.append(d_loss)

            mean_loss_each = np.mean(epoch_disc_loss_s)

        epoch_disc_loss.append(mean_loss_each)  # document all loss values

        for _ in range(k_g):
            z_sample = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = G_train.train_on_batch(z_sample, None)

        epoch_gen_loss.append(g_loss)

                # Plot the progress
        if epoch % n_i == 0:
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
            print("%d [D loss: %s] [G loss: %s]" % (epoch, mean_loss_each, g_loss))  # print loss values.
            path = save_file + '/display_%s.png'
            sample(path % epoch, G, channel, z_dim, n)
            G.save(save_file + '/generator_model_%s_z_%s.h5' % (epoch, z_dim))
            G_train.save(save_file + '/generator_train_model_%s_z_%s.h5' % (epoch, z_dim))
            D_train.save(save_file + '/discriminator_train_model_%s_z_%s.h5' % (epoch, z_dim))

    fig, ax = plt.subplots()
    epoch_disc_loss = np.array(epoch_disc_loss)
    epoch_gen_loss = np.array(epoch_gen_loss)
    plt.plot(epoch_disc_loss, label='Discriminator')
    plt.plot(epoch_gen_loss, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(save_file + '/epoch_loss.jpg', bbox_inches='tight')
    plt.close('all')
