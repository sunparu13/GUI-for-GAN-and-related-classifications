def fid_calc(file, real):
    #from __future__ import absolute_import
    #from __future__ import division
    #from __future__ import print_function

    import numpy as np
    import tensorflow as tf 
    import cv2
    import matplotlib.pyplot as plt
    import glob
    import pickle
    import os
    from keras import layers
    from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    import sys
    import time
    import functools
    from six.moves import xrange  # pylint: disable=redefined-builtin

    from sklearn.utils import shuffle
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import functional_ops

    tfgan = tf.contrib.gan


    # Shortcuts for later.
    queues = tf.contrib.slim.queues
    layers = tf.contrib.layers
    ds = tf.contrib.distributions
    framework = tf.contrib.framework

    #import fake image dataset
    fake = file
    print(real.shape)
    print(fake.shape)


    def input_stack(x_input):    #stack gray images to 3 channels, Inception V3 needs this format
        x_group = []
        for k in range(len(x_input)):
            resized = x_input[k]
            stack_3 = np.zeros((resized.shape[0], resized.shape[1], 3), "uint8")
            stack_3[:, :, 0], stack_3[:, :, 1], stack_3[:, :, 2] = resized, resized, resized
            x_group.append(stack_3)
        x_group = np.array(x_group)
        return x_group

    if real.shape[-1]!=3:    # if generated images are grayscale images

        image_real = input_stack(real)
        image_real = np.moveaxis(image_real,-1,1) #the format sequence
        fake = np.squeeze(fake, axis=-1)
        image_fake = input_stack(fake)   
        image_fake = np.moveaxis(image_fake,-1,1) #the format sequence

        print(image_real.shape)
        print(image_fake.shape)

    else:

      # if generated images are rgb images
        image_real = np.moveaxis(real,-1,1)
        image_fake = np.moveaxis(fake,-1,1)

        print(image_real.shape)
        print(image_fake.shape)
    #special row order for tf.contrib.gan function


    # In[6]:


    from sklearn.utils import shuffle
    image_real = shuffle(image_real)
    image_fake = shuffle(image_fake)


    # In[7]:


    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import functional_ops

    tfgan = tf.contrib.gan

    session = tf.InteractiveSession()

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 32

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    activations1 = tf.placeholder(tf.float32, [None, None], name = 'activations1')
    activations2 = tf.placeholder(tf.float32, [None, None], name = 'activations2')
    fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

    def inception_activations(images = inception_images, num_splits = 1):
        images = tf.transpose(images, [0, 2, 3, 1])
        size = 299
        images = tf.image.resize_bilinear(images, [size, size])
        generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
        activations = functional_ops.map_fn(
            fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'pool_3:0'),
            elems = array_ops.stack(generated_images_list),
            parallel_iterations = 1,
            back_prop = False,
            swap_memory = True,
            name = 'RunClassifier')
        activations = array_ops.concat(array_ops.unstack(activations), 0)
        return activations

    activations =inception_activations()

    def get_inception_activations(inps):
        n_batches = inps.shape[0]//BATCH_SIZE
        act = np.zeros([n_batches * BATCH_SIZE, 2048], dtype = np.float32)
        for i in range(n_batches):
            inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
            act[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = activations.eval(feed_dict = {inception_images: inp})
        return act

    def activations2distance(act1, act2):
         return fcd.eval(feed_dict = {activations1: act1, activations2: act2})

    def get_fid(images1, images2):
        assert(type(images1) == np.ndarray)
        assert(len(images1.shape) == 4)
        assert(images1.shape[1] == 3)
        assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
        assert(type(images2) == np.ndarray)
        assert(len(images2.shape) == 4)
        assert(images2.shape[1] == 3)
        assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
        assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
        print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
        start_time = time.time()
        act1 = get_inception_activations(images1)
        act2 = get_inception_activations(images2)
        fid = activations2distance(act1, act2)
        print('FID calculation time: %f s' % (time.time() - start_time))
        return fid


    # # FID Check
    #real vs fake
    images1 = image_real
    images2 = image_fake
    fid = get_fid(images1, images2) #calculate the FID score

    return fid

    #print(fid) #print the result. The lower the better