"""
@author: Jeremy Harris
@date: 01/26/2021
"""
#########################################################################
## This code loads the models that were trained with VGG16 imagenet data
## and then fed into the models (random forest, xgboos & svm). In this code
## you set your working directory, input and output folders and model
## and then the code will iterate over all file in the input folder,
## perform the necessary pre-processing, segment the images and then save
## the image to the output folder.
#########################################################################

from PIL import Image
import numpy as np
from numpy import asarray
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
import time
from keras.models import load_model
import xgboost

#set working directory on your machine:
os.chdir("/root/mlhome/segmentation/")

#####################################
######       Load Models       ######
#####################################

#Load Models
#RF_tuned_name = 'RF_tuned2.sav'
#XG_tuned_orig = 'XG_tuned.sav'
XG_tuned_name = 'XG_base_3NoPCA.sav'
#XG_tuned_gpu_name = 'XG_gpu_tuned.sav'

#RF_tuned_model = pickle.load(open(RF_tuned_name, 'rb'))
#XG_tuned_orig = pickle.load(open(XG_tuned_orig, 'rb'))
XG_tuned_model = pickle.load(open(XG_tuned_name, 'rb'))
#XG_tuned_GPU_model = pickle.load(open(XG_tuned_gpu_name, 'rb'))

#assign best params to tuned models
#RF_tuned_best = RF_tuned_model #this model was explicitely set, no need for best
#XG_tuned_orig = XG_tuned_orig.best_estimator_
XG_tuned_best = XG_tuned_model
#XG_tuned_GPU = XG_tuned_GPU_model.best_estimator_

####### YOU HAVE TO SET THIS ######
#model to use
#all options are below, uncomment the one you want to use
#pred_model = RF_tuned_best  
#pred_model = XG_tuned_orig
pred_model = XG_tuned_best   

#only use gpu model if you have a gpu -- this is just using gpu for prediction
#pred_model = XG_tuned_GPU   

#####################################
######  Prep Images to Segment ######
#####################################

#input folder path where images are stored
new_images_in = './new_images_in/' #must have / at the end

#output folder path
pred_images_out = './out_pred_images/' #must have / at the end

#load pca weights for pre-processing of new images
with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)

#crop image dimensions 
### !! THIS MUST MATCH WHAT WAS DONE FOR TRAINING MODELS !! ###
left = 0
top = 165
right = 2262
bottom = 530

#get all files names in folder to iterate through
files_in_folder = os.listdir(new_images_in)

#set shape of new images to revert back to after processing
img_shape = Image.open(new_images_in+files_in_folder[0])
img_shape = img_shape.crop((left, top, right, bottom))
img_shape = asarray(img_shape)
img_shape = img_shape.shape[-3:-1]

#get weights from VGG16 model and assign to new_model (previously saved)
vgg_model = load_model('new_model.h5')

#####################################
######  Segment and Save Images #####
#####################################

#load, process, predict, reshape and save each image
t1=time.time()
for img in files_in_folder:
    image = Image.open(new_images_in+"00001.jpg") #read in 1 image at a time
    cropped = image.crop((left, top, right, bottom)) #crop image
    img_array = asarray(cropped) #convert to array
    new_img = np.expand_dims(img_array, axis=0) #shape for VGG16
    new_img = vgg_model.predict(new_img) #get VGG16 weights
    new_features = new_img.reshape(-1, new_img.shape[3]) #shape
    #new_norm = preprocessing.normalize(new_features) #normalize
    #new_pca = pca.transform(new_norm)
    
    #predict with chosen model (above -- uncomment model) here
    new_preds = pred_model.predict(new_features)
    
    #shape into image file
    new_pred_img = new_preds.reshape(img_shape)
    
    #save image to chosen folder (above)
    plt.imsave(pred_images_out+img, new_pred_img, cmap='gray')
t2=time.time()
print("Time to predict "+str(len(files_in_folder)) +" images: " + str(round(t2-t1))+" seconds") 

