
def tsne_visual(file, real):

    import cv2
    import glob
    import sys
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    dcgan = file    #load data from the saved npy file
    if dcgan.shape[-1] != 3:
        dcgan = np.squeeze(dcgan, axis=-1)

    print(dcgan.shape)
    print(real.shape)


    def load_data(x):
        image_data = []

        for i in range(len(x)):
            image_data.append(x[i].flatten())  #should flatten the data into vector
        X= np.array(image_data)

        return X
    dcgan = load_data(dcgan)
    real=load_data(real)

    print(dcgan.shape)
    print(real.shape)

    #labelize the dataset
    label_dc = np.ones(len(dcgan)).astype(np.int)   #all 1
    label = np.zeros(len(real)).astype(np.int) #all 0

    total = np.concatenate((dcgan, real), axis = 0)  #combine the subset together and form a total dataset with label
    total_l = np.concatenate((label_dc, label), axis = 0)
    print(total.shape, total_l.shape)


   #get_ipython().run_line_magic('matplotlib', 'inline')
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2).fit(total)   #reduce to 2d-space

    X_em = tsne.fit_transform(total)  #fit the transform

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    for i in range(0, X_em.shape[0]):
        if total_l[i] == 0:
            c1 = plt.scatter(X_em[i, 0], X_em[i, 1], c='r', marker='+')
        if total_l[i] == 1:
            c2 = plt.scatter(X_em[i, 0], X_em[i, 1], c='g', marker='+')

    plt.legend([c1, c2], ['real', 'gan' ])
    plt.axis('off')
    # plt.xlim(-20,25)
    # plt.ylim(-20,25)
    plt.savefig('tsne.jpg', bbox_inches = 'tight')




    # In[ ]:


    # dimension reduction to 3d-space

    #get_ipython().run_line_magic('matplotlib', 'notebook  # can operate the result diagram.')

    #from sklearn.manifold import TSNE

    #tsne = TSNE(n_components=3).fit(total)

    #X_em = tsne.fit_transform(total)

    # plot
    from mpl_toolkits.mplot3d import Axes3D

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #for i in range(0, X_em.shape[0]):
     #   if total_l[i] == 0:    #label 0 represents the generated images
      #      c1 = ax.scatter(X_em[i,0],X_em[i,1],X_em[i,2],c='r',    s=50,marker='+')
       # if total_l[i] == 1:  #label 1 represents the real images
        #    c2 = ax.scatter(X_em[i,0],X_em[i,1],X_em[i,2],c='g',    s=50,marker='o')
        #if total_l[i] == 1:  #label 1 represents the real images
         #   c3 = ax.scatter(X_em[i,0],X_em[i,1],X_em[i,2],c='b',    s=50,marker='o')
  #  plt.legend([c1, c2, c3], ['real','dcgan','wgan'])
    #plt.xlim(-80,80)
    #plt.ylim(-80,80)
   # plt.show()





