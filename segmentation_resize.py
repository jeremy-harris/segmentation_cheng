"""
@author: Jeremy Harris
@date: 12/3/2020
"""
#########################################################################
### This code combines the CNN pre_trained model VGG16 with the imagenet
### data, pulls the weights off of it and then feeds it into an RF model
### for prediction of pixel level semantic segmentation. This is useful
### when we don't have a large train set (marking image masks) and we don't
### need to train a custom model. The code was written referencing and using
### example code from: github.com/bnsreenu (159b)
#########################################################################

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Model
import os
from keras.applications.vgg16 import VGG16
from time import process_time

#set resize information to get images smaller. Number is % of original size
scale_pct = 70


#set path to images
train_path = "./train_frames/"
mask_path = "./train_masks/"
test_path = "./test_frames/"

####################
#setup list for storing training images
train_images = []

#get training images
os.chdir("/root/mlhome/segmentation/")
for i in sorted(glob.glob(os.path.join(train_path, "*.tif"))):
    img = cv2.imread(i, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_w = int(img.shape[1] * scale_pct / 100) #scale image
    img_h = int(img.shape[0] * scale_pct / 100) #scale image
    
    dsize_i = (img_w, img_h)
    img_out = cv2.resize(img, dsize_i)
    
    train_images.append(img_out)

#Convert list to array for machine learning processing        
train_images = np.array(train_images)


####################
#setup list for storing training masks
train_masks = []

#get training masks
os.chdir("/root/mlhome/segmentation/")
for m in sorted(glob.glob(os.path.join(mask_path, "*.tif"))):
    mask = cv2.imread(m, 0)
    mask_w = int(mask.shape[1] * scale_pct / 100) #scale image
    mask_h = int(mask.shape[0] * scale_pct / 100) #scale image
    
    dsize_m = (mask_w, mask_h)
    mask_out = cv2.resize(mask, dsize_m)
    
    train_masks.append(mask_out)

#Convert list to array for machine learning processing        
train_masks = np.array(train_masks)

#set x & y up for standard ML syntax
#X_train = tf.keras.applications.mobilenet.preprocess_input(train_images)
#y_train = tf.keras.applications.mobilenet.preprocess_input(train_masks)
X_train = train_images
y_train = train_masks
######################
#Load VGG16 model without classifier (we'll use RF for that)
#include_top=False removes the dense layers for prediction
#weights come from the pretrained imagenet models

#set image size
IMG_W = X_train.shape[2]
IMG_H = X_train.shape[1]

VGG_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_H, IMG_W, 3))

#set the layers from the model as non-trainable layers because we just
#want to pull the weights from the model - not actually train it as it
#has already been trained on imagenet data
for layer in VGG_model.layers:
    layer.trainable=False
#VGG_model.summary()

#after block1_conv1 the image dimensions change from our image dimensions
#we'll pull only the block1_conv1 layer which still gives 64 trainable features
new_model = Model(
        inputs=VGG_model.input,
        outputs=VGG_model.get_layer('block1_conv2').output)
#new_model.summary()

######################
#extract features from weighted model
features = new_model.predict(X_train)

'''
#Plot features to view them
square = 8
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()
'''
X = features
#combine all pixels into rows for feature columns - we need to do this so 
#that we can feed this into RF model
X = X.reshape(-1, X.shape[3]) 

#reshape Y to match X
Y = y_train.reshape(-1)

###Here we want to drop pizels with a value of 0 as it is the background and
###not worth wasting resources on to detect.

#Create df to use for RF - add Y as our label for this train data
df = pd.DataFrame(X)
df['Label'] = Y

#Look to verify lables are correct (i.e. not all blank)
#print(df['Label'].unique())
#print(df['Label'].value_counts())

#Drop pixels with value of 0 --
df2 = df[df['Label'] != 0]

#######################
#Random Forest
#set X &y from df for RF
X_RF = df2.drop(labels=['Label'], axis=1)
Y_RF = df2['Label']

import random
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(30)
model = RF(n_estimators = 50, random_state = 42)

# Train the model on training data
model.fit(X_RF, Y_RF) 

#test_result = model.score(X_RF, Y_RF)
#print(test_result)


#Save model for future use
filename = 'RF_model2.sav'
pickle.dump(model, open(filename, 'wb'))

''''
### Try using RF and GPU using cuML package
from cuml import RandomForestClassifier as cuRF



'''
#Load model.... 
loaded_model = pickle.load(open(filename, 'rb'))
#loaded_model = model

#start time for process
t1_start = process_time()

#Test on a different image
for i in sorted(glob.glob(os.path.join(test_path, "*.jpg"))):
    test_img_in = i

test_img = cv2.imread(test_img_in, 1)    
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
       
#resize images to match
test_w = int(test_img.shape[1] * scale_pct / 100) #scale image
test_h = int(test_img.shape[0] * scale_pct / 100) #scale image
    
dsize_t = (test_w, test_h)
test_out = cv2.resize(test_img, dsize_t)
    
test_out = np.expand_dims(test_out, axis=0)

#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
X_test_feature = new_model.predict(test_out)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])

prediction = loaded_model.predict(X_test_feature)

#change pixel values to stand out better
#post_image = prediction
#post_image[post_image == 1] = 100


#get shape of mask to match Y value so we can recreate the same size image
new_shape = train_masks.shape[-2:]

#View and Save segmented image
prediction_image = prediction.reshape(new_shape)
plt.imshow(prediction_image, cmap='gray')

t1_stop = process_time()
print("Elapsed time in seconds: ", t1_stop - t1_start)

#show cool/warm image
post_image_plt = post_image.reshape(new_shape)
plt.imshow(post_image_plt, cmap = 'coolwarm')

#save images
plt.imsave('resized.jpg', prediction_image, cmap='gray')
plt.imsave('resized_coolwarm.jpg', prediction_image, cmap='coolwarm')

