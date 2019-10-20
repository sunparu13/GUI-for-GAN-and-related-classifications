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

#total_train = np.concatenate((p_train, p_cda), axis=0)


#real_dim=(96,96,channel) #the input dimension of the discriminator
#save_file = 'dcgan_result/e1'  #save the result in the subfolder ex1, ex1 means experiment Nr.1

# generator variant1, named 'gen1', can be used in both dcgan and wgan-div
def gan_g1(z_dim, channel_num):

    model = Sequential()

    model.add(Dense(128 * 12 * 12, activation="relu", input_dim=z_dim))     #input:(1000, None), output: one layer with 128*12*12=18432 neuron
    model.add(Reshape((12, 12, 128))) #reshape to 128 feature maps, each dimension is 12*12
    model.add(BatchNormalization(momentum=0.8))  #BN layer
    model.add(UpSampling2D())           #upscale to 24*24*128
    model.add(Conv2D(64, kernel_size=4, padding="same"))   #first conv layer, width and height are notchanged due to 'same' padding. output size is 24*24*64, 64 is the number of the filter, and also the depth of the output
    model.add(BatchNormalization(momentum=0.8))  #BN layer
    model.add(Activation("relu"))    #ReLU activation
    model.add(UpSampling2D())       #upsampling to 48*48*64, principle is to repeat the rows and columns of the data by size 2 and 2 as default.
    model.add(Conv2D(32, kernel_size=4, padding="same")) #second conv layer, output size is 48*48*32
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(16, kernel_size=4, padding="same"))  #third conv layer, output size is 48*48*16
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D()) #output size: 96*96*16
    model.add(Conv2D(channel_num, kernel_size=4, padding="same"))    #fourth conv layer, output size is 96*96*channel_num, which means a kernel filter with the size 4*4*16 is applied
    model.add(Activation("tanh"))   #tanh activation function, output is between -1 to 1, which is the generated image
    #plot_model(model, show_shapes=True, to_file='gen1.png')
    model.summary()
    noise = Input(shape=(z_dim,))
    img = model(noise)
    return Model(noise, img)  #establish the model with input 'noise' and output 'img'

# generator variant2, named 'gen2', can be used in both dcgan and wgan-div
def gan_g2(z_dim, channel_num):

    model = Sequential()

    model.add(Dense(128 * 12 * 12, activation="relu", input_dim=z_dim)) #input:(1000, None), output: one layer with 128*12*12=18432 neurons
    model.add(Reshape((12, 12, 128))) #reshape to 128 feature maps, each dimension is 12*12
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(64,(4,4),strides=(2, 2),padding='same',  #deconvolutional layer with stride 2 , which the width and height is now 2x, kernel filter number is 64, so output size is 24*24*64
                        kernel_initializer='glorot_uniform'))  #initialization the weights using Xavier Apporoach
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(32,(4,4),strides=(2, 2),padding='same',
                        kernel_initializer='glorot_uniform'))  #output size is 48*48*32.
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(channel_num,(4,4),strides=(2, 2),padding='same', #output size is 96*96*channel_num.
                        kernel_initializer='glorot_uniform'))
    model.add(Activation("tanh")) #tanh activation function, output is between -1 to 1, which is the generated image
    noise = Input(shape=(z_dim,))
    img = model(noise)
    return Model(noise, img) #establish the model with input 'noise' and output 'img'

# generator variant3, named 'gen3', can be used in both dcgan and wgan-div
def gan_g3(z_dim, channel_num):

    model = Sequential()

    model.add(Dense(256 * 6 * 6, activation="relu", input_dim=z_dim)) #input:(1000, None), output: one layer with 256*6*6=9216 neurons
    model.add(Reshape((6, 6, 256)))  #6*6*256
    model.add(UpSampling2D()) #12*12*256
    model.add(Conv2D(128, kernel_size=4, padding="same"))  #output :12*12*128, filter size: 4*4*256, filter number:128
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D()) #output: 24*24*128
    model.add(Conv2D(64, kernel_size=4, padding="same")) #output :24*24*64, filter size: 4*4*128, filter number:64
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D()) #output :48*48*64
    model.add(Conv2D(32, kernel_size=4, padding="same")) #output :48*48*32, filter size: 4*4*64, filter number:32
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D()) #output :96*96*64
    model.add(Conv2D(channel_num, kernel_size=4, padding="same")) #output :96*96*channel_num, filter size: 4*4*64, filter number:1
    model.add(Activation("tanh")) #tanh activation function, output is between -1 to 1, which is the generated image
    noise = Input(shape=(z_dim,))
    img = model(noise)
    return Model(noise, img) #establish the model with input 'noise' and output 'img'


def resnet_gen(z_dim, channel):
    """Resnet Generator"""

    def residual_block(layer_input, df):
        """Residual block described in paper"""
        d = BatchNormalization(momentum=0.8)(layer_input)
        d = Activation('relu')(d)
        d = Conv2D(df, kernel_size=4, strides=1, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = Activation('relu')(d)
        d = Conv2D(df, kernel_size=4, strides=1, padding='same')(d)
        d = Add()([d, layer_input])  # connect the input layer to the output
        return d

    # Image input
    z_in = Input(shape=(z_dim,))
    z = z_in

    z = Dense(64 * 12 * 12, activation="relu", input_dim=z_dim)(
        z)  # input:(1000, None), output: one layer with 128*12*12=18432 neurons
    r = Reshape((12, 12, 64))(z)  # reshape to 128 feature maps, each dimension is 12*12

    for _ in range(3):  # 3 resbolcks
        r = residual_block(r, 64)
        r = UpSampling2D()(r)

    # Propogate signal through residual blocks
    r = residual_block(r, 64)
    r = BatchNormalization(momentum=0.8)(r)
    r = Activation('relu')(r)
    r = Conv2D(channel, kernel_size=4, padding='same', activation='tanh')(r)

    g_model = Model(z_in, r)
    g_model.summary()
    return g_model



def gan_g5(z_dim, channel_num):

    model = Sequential()

    model.add(Dense(1024 * 6 * 6, activation="relu", input_dim=z_dim)) #output: one layer with 1024*6*6 neurons
    model.add(Reshape((6, 6, 1024))) #reshape to 1024 feature maps, each dimension is 6*6
    #model.add(Activation("relu"))
    model.add(Conv2DTranspose(512,(4,4),strides=(2, 2),padding='same',  #deconvolutional layer with stride 2 , which the width and height is now 2x, kernel filter number is 64, so output size is 12*12*512
                        kernel_initializer='glorot_uniform'))  #initialization the weights using Xavier Apporoach  # kernel_initializer='glorot_uniform'
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(256,(4,4),strides=(2, 2),padding='same',
                        kernel_initializer='glorot_uniform'))  #output size is 24*24*256.
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(128,(4,4),strides=(2, 2),padding='same',
                        kernel_initializer='glorot_uniform'))  #output size is 48*48*128.
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(channel_num,(4,4),strides=(2, 2),padding='same', #output size is 96*96*channel_num.
                        kernel_initializer='glorot_uniform'))
    model.add(Activation("tanh")) #tanh activation function, output is between -1 to 1, which is the generated image
    model.summary()
    noise = Input(shape=(z_dim,))
    img = model(noise)
    return Model(noise, img) #establish the model with input 'noise' and output 'img'

def resnet_gen2(z_dim, channel_num):
    def identity_block(layer_input, filters):
        f1, f2 = filters

        X_s = layer_input

        d = Conv2DTranspose(f1, kernel_size=1, strides=1, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = Activation('relu')(d)
        d = Conv2DTranspose(f2, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, X_s])  # connect the input layer to the output
        d = Activation('relu')(d)
        return d

    def conv_block(layer_input, filters):
        X_s = layer_input

        f1, f2 = filters

        d = Conv2DTranspose(f1, kernel_size=1, strides=2, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = Activation('relu')(d)
        d = Conv2DTranspose(f2, kernel_size=4, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)

        X_s = Conv2DTranspose(f2, kernel_size=1, strides=2, padding='same')(X_s)
        X_s = BatchNormalization(momentum=0.8)(X_s)

        d = Add()([d, X_s])  # connect the input layer to the output
        d = Activation('relu')(d)
        return d

    # Image input
    z_in = Input(shape=(z_dim,))
    z = z_in

    z = Dense(1024 * 6 * 6, activation="relu", input_dim=z_dim)(z)  # input:(1000, None), output: one layer with 128*12*12=18432 neurons
    r = Reshape((6, 6, 1024))(z)  # reshape to 128 feature maps, each dimension is 12*12

    r = Conv2DTranspose(512, kernel_size=4, strides=1, padding='same')(r)
    r = BatchNormalization(momentum=0.8)(r)
    r = Activation("relu")(r)

    r = conv_block(r, [512, 256])
    r = identity_block(r, [256, 256])

    r = conv_block(r, [256, 128])
    r = identity_block(r, [128, 128])

    r = conv_block(r, [128, 64])
    r = identity_block(r, [64, 64])

    r = conv_block(r, [64, 32])
    r = identity_block(r, [32, 32])

    r = Conv2D(channel_num, kernel_size=4, padding='same', activation='tanh')(r)

    g_model = Model(z_in, r)

    return g_model


# DCGAN discriminator variant2,named dis2
def dcgan_d1(input_shape_d):

    model = Sequential()

    model.add(GaussianNoise(stddev=0.02))   #add some noise to the first layer, with std=0.02
    model.add(Conv2D(32, kernel_size=5, strides=2, input_shape= input_shape_d, padding="same", kernel_initializer='glorot_uniform'))
    #suppose input size is  96*96*1, than output size is: 48*48*32
    #kernel_initializer= RandomNormal(mean=0.0, stddev=0.2) can be added be selected
    model.add(LeakyReLU(alpha=0.2)) #leaky ReLu activation function with the gradient 0.2 for the negative part. kernel_initializer=RandomNormal(mean=0.0, stddev=0.1)
    model.add(Dropout(0.25)) #drop 25% neurons randomly
    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same", kernel_initializer='glorot_uniform')) # output size is 24*24*64, with 64 kernel filter, each one with size 5*5*32
    model.add(BatchNormalization()) #BN layer
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer='glorot_uniform')) # output size is 12*12*128, with 128 kernel filter, each one with size 3*3*64
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer='glorot_uniform')) # without upscaling because stride is 1.  output size is 12*12*256, with 256 kernel filter, each one with size 3*3*128
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten()) #flatten layer as the input of FCN layer, neuron number is 12*12*256=36864
    model.add(Dense(1, activation='sigmoid')) #output is only one to show the probability.
    input_d = Input(shape=input_shape_d)
    output_d = model(input_d)
    return Model(input_d, output_d)

# DCGAN discriminator variant2,named dis2
def dcgan_d2(input_shape_d):

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
    model.add(Dense(1, activation='sigmoid')) #output is only one to show the probability being classified as the real image.
    input_d = Input(shape=input_shape_d)
    output_d = model(input_d)
    return Model(input_d, output_d)


def resnet_dis(channel_n):
    def identity_block(layer_input, filters):
        f1, f2 = filters

        X_s = layer_input

        d = Conv2D(f1, kernel_size=1, strides=1, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(f2, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, X_s])  # connect the input layer to the output
        d = LeakyReLU(alpha=0.2)(d)
        return d

    def conv_block(layer_input, filters):
        X_s = layer_input

        f1, f2 = filters

        d = Conv2D(f1, kernel_size=1, strides=2, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(f2, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)

        X_s = Conv2D(f2, kernel_size=(1, 1), strides=2, padding='same')(X_s)
        X_s = BatchNormalization(momentum=0.8)(X_s)

        d = Add()([d, X_s])  # connect the input layer to the output
        d = LeakyReLU(alpha=0.2)(d)
        return d

    # Image input
    z_in = Input(shape=(96, 96, channel_n))
    z = z_in

    r = Conv2D(32, kernel_size=4, strides=2, padding='same')(z)
    r = BatchNormalization(momentum=0.8)(r)
    r = LeakyReLU(alpha=0.2)(r)

    r = conv_block(r, [32, 64])
    r = identity_block(r, [32, 64])

    r = conv_block(r, [64, 128])
    r = identity_block(r, [64, 128])

    r = conv_block(r, [128, 256])
    r = identity_block(r, [128, 256])

    r = AveragePooling2D()(r)
    r = Flatten()(r)
    r = Dense(1, activation='sigmoid')(r)

    g_model = Model(z_in, r)

    return g_model


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
def train(X, G, D, loops, batch_size, k_d, k_g, index_r, index_f, opt_g, opt_d, z_dim, channel, save_file, n_i, n):

    import matplotlib.pyplot as plt

    G_train = combined_g(G, D, z_dim)  #or gen2, dis1
    G_train.compile(loss='binary_crossentropy', optimizer=opt_g)
    D_train = D
    D_train.compile(loss='binary_crossentropy', optimizer=opt_d)

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
            d_loss_real = D_train.train_on_batch(imgs, real_smooth)  #trick: seperate train real image batch and fake imahe batch
            d_loss_fake = D_train.train_on_batch(gen_imgs, fake_smooth)
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


def train2(X, G, D, nb_epochs, batch_size, k_d, k_g, index_r, index_f, opt_g, opt_d, z_dim, channel, save_file, n_i, n):

    import matplotlib.pyplot as plt

    G_train = combined_g(G, D, z_dim)  # or gen2, dis1
    G_train.compile(loss='binary_crossentropy', optimizer=opt_g)
    D_train = D
    D_train.compile(loss='binary_crossentropy', optimizer=opt_d)

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