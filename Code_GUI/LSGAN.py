from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Dropout, ZeroPadding2D,Reshape, Input, Add, GaussianNoise, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as K
import cv2
import glob
import imageio
import sys
import os
import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.initializers import glorot_uniform


#GAN combined (Discriminator is fixed)
def combined_g(G, D, z_dim):
    frozen_D = Model(
            inputs=D.inputs,
            outputs=D.outputs)    #freeze the discriminator when training generator
    frozen_D.trainable = False
    z = Input(shape=(z_dim,))
    img = G(z)
    valid = frozen_D(img)
    gan = Model(z, valid)  #establish the training generative model with input 'noise' and output 'img' and frozen D
    gan.summary()
    return gan


#form 10x10 connected images, each of them is 96x96
def sample(path,G, channel, z_dim, n):
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

# training process variant 1
def train5(X, G, D, loops, batch_size, k_d, k_g, index_r, index_f, opt_g, opt_d, z_dim, channel, save_file, n_i, n):

    import matplotlib.pyplot as plt

    G_train = combined_g(G, D, z_dim)  #or gen2, dis1
    G_train.compile(loss='mse', optimizer=opt_g)
    D_train = D
    D_train.compile(loss='mse', optimizer=opt_d)

    lossesd = []
    lossesg = []

    # Rescale image from [0,255] to [-1 to 1]
    X = (X - 127.5)/127.5
    if X.shape[-1]!=3:
        X = np.expand_dims(X, axis=3)  #if gray images, than add a dimension with the channel nummber 1.

    # Labels: real:1, fake:0

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Labels variant : a trick is to smmoth the label to be real:0.9, fake:0.1
    real_smooth = np.ones((batch_size, 1))*index_r
    fake_smooth = np.ones((batch_size, 1))*index_f

    # Training Process
    for loop in range(loops+1):  #each loop, just train a randomly selected batch instead of training batches that read through the entire dataset.
        for _ in range(k_d):
            #  Train Discriminator k times
            idx = np.random.randint(0, X.shape[0], batch_size)  #random select batch size real images
            imgs = X[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, z_dim))   #sample noise 0-1 uder a normal distribution
            gen_imgs = G.predict(noise)  #generate batch of images

            # Train the discriminator (real classified as 1 and generated as 0)
            d_loss_real = D_train.train_on_batch(imgs, real)  #trick: seperate train real image batch and fake imahe batch
            d_loss_fake = D_train.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #calculate average discriminator loss

        for _ in range(k_g):
            noise = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = G_train.train_on_batch(noise, real)  #train fake image batch with GAN model with frozen D.

        lossesd.append(d_loss)
        lossesg.append(g_loss) #document all loss values

        if loop % n_i == 0:
            print("%d [D loss: %f] [G loss: %f]" % (loop, d_loss, g_loss))
            G.save(save_file + '/generator_model_%s_z_%s.h5' % (loop, z_dim))
            G_train.save(save_file + '/generator_train_model_%s_z_%s.h5' % (loop, z_dim))
            D_train.save(save_file + '/discriminator_train_model_%s_z_%s.h5' % (loop, z_dim))
            path = save_file + '/display_%s.png'
            sample(path %loop, G, channel, z_dim, n)


    fig, ax = plt.subplots()
    lossesd = np.array(lossesd)
    lossesg = np.array(lossesg)
    plt.plot(lossesd, label='Discriminator')
    plt.plot(lossesg, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(save_file + '/loop_loss.jpg', bbox_inches='tight')
    plt.close('all')

#beform running, we should change the path manually
# X is the real image dataset, D_train is the compiled D model, G_train is the compiled GAN model with frozen G


def train6(X, G, D, nb_epochs, batch_size, k_d, k_g, index_r, index_f, opt_g, opt_d, z_dim, channel, save_file, n_i, n):

    import matplotlib.pyplot as plt

    G_train = combined_g(G, D, z_dim)  # or gen2, dis1
    G_train.compile(loss='mse', optimizer=opt_g)
    D_train = D
    D_train.compile(loss='mse', optimizer=opt_d)

    X = shuffle(X)
    # Rescale image from [0,255] to [-1 to 1]
    X = (X - 127.5) / 127.5
    if X.shape[-1] != 3:
        X = np.expand_dims(X, axis=3)  # if gray images, than add a dimension with the channel nummber 1.

    # Labels: real:1, fake:0
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Labels variant : a trick is to smooth the label to be real:0.9, fake:0.1
    real_smooth = np.ones((batch_size, 1)) * index_r
    fake_smooth = np.ones((batch_size, 1)) * index_f

    for epoch in range(nb_epochs+1):

        nb_batches = int(X.shape[0] / batch_size)

        epoch_gen_loss = []
        epoch_disc_loss_s =[]
        epoch_disc_loss = []

        for _ in range(k_d):

            for index in range(nb_batches):

                # generate a new batch of noise
                noise = np.random.normal(0, 1, (batch_size, z_dim))

                # get a batch of real images
                image_batch = X[index * batch_size:(index + 1) * batch_size]  # go through batch to batch

                # generate a batch of fake images
                gen_img_batch = G.predict(noise)

                d_loss_real = D_train.train_on_batch(image_batch,
                                                     real_smooth)  # trick: seperate train real image batch and fake imahe batch
                d_loss_fake = D_train.train_on_batch(gen_img_batch, fake_smooth)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # calculate average discriminator loss
                epoch_disc_loss_s.append(d_loss)

            mean_loss_each = np.mean(epoch_disc_loss_s)

        epoch_disc_loss.append(mean_loss_each)  # document all loss values

        for _ in range(k_g):
            noise = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = G_train.train_on_batch(noise, real)

        epoch_gen_loss.append(g_loss)



        # If at save interval => save generated image samples
        # the save_interval can be tuned and the generated images after different training epochs can be used for quality evaluation with IS /FID


        if epoch % n_i == 0:
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
            print("%d [D loss: %f] [G loss: %f]" % (epoch, mean_loss_each, g_loss))  # print loss values.
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
    plt.savefig(save_file +'/epoch_loss.jpg', bbox_inches='tight')
    plt.close('all')