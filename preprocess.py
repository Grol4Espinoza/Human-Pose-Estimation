##!python
#!/usr/bin/env python

import os
import shutil
from scipy.io import loadmat
import numpy as np
import scipy.io
from operator import itemgetter
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2 
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras
from keras.optimizers import RMSprop
from tensorflow.keras.layers import (
                                    Dense,
                                    Conv2D, 
                                    BatchNormalization, 
                                    ReLU, 
                                    Add,
                                    Input,
                                    MaxPooling2D,
                                    UpSampling2D,
                                    )
from keras.models import Model
from keras.losses import mean_squared_error
from math import exp
from sklearn.model_selection import train_test_split
%matplotlib inline
############################################################################
def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])             
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))                

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:            
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields:
                field = [field]
                ret[field_name] = field

    else:
        ret = obj

    return ret
############################################################################
def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])             
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))                

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:            
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields:
                field = [field]
                ret[field_name] = field

    else:
        ret = obj

    return ret

############################################################################
def print_dataset_obj(obj, depth = 0, maxIterInArray = 20):
    prefix = "  "*depth
    if type(obj) == dict:
        for key in obj.keys():
            print("{}{}".format(prefix, key))
            print_dataset_obj(obj[key], depth + 1)
    elif type(obj) == list:
        for i, value in enumerate(obj):
            if i >= maxIterInArray:
                break
            print("{}{}".format(prefix, i))
            print_dataset_obj(value, depth + 1)
    else:
        print("{}{}".format(prefix, obj))
############################################################################
def return_image_joints(name,data):
    for item in data: # guardar coordenadas de los joints
        if item[0] == name:
            #print(item[1]) 
            return item[1]
############################################################################
rightconnections = [
                    (0,1),(1,2),(3,4),(4,5),(2,6),
                    (3,6),(6,7),(7,8),(8,9),(10,11),
                    (11,12),(12,7),(13,7),(13,14),(14,15)
                   ]
size_img_x = 256
size_img_y = 256
def draw_img_joints(file_name, data, resize = False ):    
    # Load image
    #img = cv2.imread(Path_To_Single_Person_Images + "/" + file_name,1)  
    img = image.load_img(Path_To_Single_Person_Images + "/" + file_name)
    img = image.img_to_array(img) 
    img = img/255
    if resize:
        img = np.float32(tf.image.resize(img,(size_img_x, size_img_y)))  
    pts = return_image_joints(file_name, data)        
    #plt.imshow(img)  
    X = [x[0] for x in pts]
    Y = [y[1] for y in pts]
    X = [int(x) for x in X]
    Y = [int(y) for y in Y]
    
    for i in range(16):
        for j in range(16):
            if (i,j) in rightconnections:
                if X[i]>0 and X[j]>0 and Y[i]>0 and Y[j]>0:
                    img = cv2.line(img,(X[i],Y[i]),(X[j],Y[j]),(1,0,0),5)
                    plt.scatter(X[i], Y[i], marker="o", color="red", s=20)
                    plt.scatter(X[j], Y[j], marker="o", color="red", s=20)
                    
    plt.imshow(img)
############################################################################
def load_image(train_data, a, b):
    train = np.asarray(train_data[a:b])
    train_image = np.zeros((b-a,size_img_x,size_img_y,3))
    for i in tqdm(range(a,b)):
        name_img = train[i][0]
        img = image.load_img(Path_To_Single_Person_Images + '/' + name_img)
        img = image.img_to_array(img)
        img_x = img.shape[1]
        img_y = img.shape[0]
        scala_x = img_x / size_img_x
        scala_y = img_y / size_img_y
        for j in range(len(train[i][1])): # escala los puntos clave
            train[i][1][j] = np.array([train[i][1][j][0] / scala_x, train[i][1][j][1] / scala_y])            
        img = tf.image.resize(img,(size_img_x, size_img_y))        
        img = img/255
        train_image[i] = img
    return train_image, train
############################################################################
def MakeHeatmap(x, y, width, height, show = False):
    # Probability as a function of distance from the center derived
    # from a gaussian distribution with mean = 0 and stdv = 1
    scaledGaussian = lambda x : exp(-(1/2)*(x**2))

    imgSize = (height, width)
    center_x = x
    center_y = y

    isotropicGrayscaleImage = np.zeros((imgSize[0],imgSize[1]),np.uint8)

    for i in range(imgSize[0]):
        for j in range(imgSize[1]):

            # find euclidian distance from center of image (x,y) 
            # and scale it to range of 0 to 2.5 as scaled Gaussian
            # returns highest probability for x=0 and approximately
            # zero probability for x > 2.5

            distanceFromCenter = np.linalg.norm(np.array([i-center_y,j-center_x]))
            #distanceFromCenter = 18*distanceFromCenter/(imgSize/2)
            scaledGaussianProb = scaledGaussian(distanceFromCenter)
            isotropicGrayscaleImage[i,j] = np.clip(scaledGaussianProb*255,0,255)   
    
    return isotropicGrayscaleImage
############################################################################    
def Joints_heatmaps(lista_de_joints, heatmap_size_x, heatmap_size_y, num_heatmaps = 16, show = False):
    heatmaps = np.zeros((16,64,64))
    for i in range(num_heatmaps):
        x, y = lista_de_joints[i] 
        x = x / 4 # entre 4 por que el array es de 256x256
        y = y / 4 # entre 4 por que el array es de 256x256
        heatmaps[i] = MakeHeatmap(x, y, heatmap_size_x, heatmap_size_y)
    if show:
        plotImages(heatmaps, num_heatmaps)
    return heatmaps
############################################################################        
def plotImages(images_arr, num_images):
    fig, axes = plt.subplots(1, num_images, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()