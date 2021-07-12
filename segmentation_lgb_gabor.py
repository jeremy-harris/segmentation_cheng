"""
@author: Jeremy Harris
@date: 01/15/2021
"""
#########################################################################
### This code combines the CNN pre_trained model VGG16 with the imagenet
### data, pulls the weights off of the model  and then feeds it into ML models
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
from keras.models import Model
import pickle
import os
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split


############################################################
#################     Load Images     ######################
############################################################

#set path to images; I had 26 images so I split them 70/30 and put the 
#image into the folder locations below. 18 training, 8 testing.

my_dir = "/root/mlhome/segmentation/segmentation_cheng/"
git_dir = "/root/mlhome/segmentation/segmentation_cheng/"
img_path = "./train_images/"
mask_path = "./train_masks/"


def get_files(c_dir, path, cv2_flag): #flag 1 = color, 0 = grayscale
    os.chdir(c_dir)
    files = []
    for i in sorted(glob.glob(os.path.join(path, "*.tif"))):
        img = cv2.imread(i, cv2_flag)
        if cv2_flag == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        files.append(img)
    files = np.array(files)
    return(files)

train_images = get_files(my_dir, img_path, 1)
train_masks = get_files(my_dir, mask_path, 0)

##########################################################
#################     VGG16 Work    ######################
##############################################array############

#Pull the VGG16 model and extract the weights from VGG16 training on imagenet
#include_top=False removes the dense layers for prediction

#set image size, this is useful if images aren't of the same size
IMG_W = train_images.shape[2]
IMG_H = train_images.shape[1]

#Pull VGG16 model with imagenet weights and match initial image size
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

### Use block 1 conv2 to get 64 features  ###
#after block1_conv2 the image dimensions change from our orig dimensions
#we'll pull only the block1_conv2 layer which still gives 64 trainable features
new_model = Model(
        inputs=VGG_model.input,
        outputs=VGG_model.get_layer('block1_conv2').output)
#new_model.summary()

#save model for later use
#new_model.save('./segmentation_cheng/new_model.h5')

##################################################
##########     Generate Features   ###############
##################################################

###Generate features (our predictors) that we will use to predict the pixel
### value when we feed the dataframe into our ML models.

def get_features(input_array):
    features_out = []
    i=0
    while i < len(input_array):
        img = np.expand_dims(input_array[i], axis=0)
        get_features = new_model.predict(img) #this is where we get VGG16 weghts
        get_features = np.squeeze(get_features) #remove the leading 1 for images
        features_out.append(get_features)
    
        i += 1

    features_out = np.array(features_out)
    #get all values into a single column for modeling
    features_out = features_out.reshape(-1, features_out.shape[3])
    return(features_out)

X1 = get_features(train_images)
X1 = pd.DataFrame(X1)


### get gabor features ###

def get_gabor(input_array, ksize):
    i=0
    df_out = pd.DataFrame()
    while i < len(input_array):
        img = input_array[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1)
        df = pd.DataFrame()
        df['orig'] = img
        num = 1  #To count numbers up in order to give Gabor features a label in the data frame
        #kernels = []

        for theta in range(2):   #Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  #Sigma with 1 and 3
                for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                    for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                            
                        gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                        #kernels.append(kernel)
                        #Now filter the image and add values to a new column 
                        fil_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        fil_img = fil_img.reshape(-1)
                        df[gabor_label] = fil_img  #Labels columns as Gabor1, Gabor2, etc.
                        print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                        num += 1  #Increment for gabor column label
        
        #img_label = 'img_'+str(i)
        df_out = df_out.append(df, ignore_index=True)
               
        i += 1
        
    return(df_out)

X2 = get_gabor(train_images, 5)

#add VGG features and gabor features together
df = pd.concat([X1, X2], axis=1)

#reshape Y to match X
Y = train_masks.reshape(-1)

#Drop pixels with value of 0 -- no need to detect the background in this case
df = df[df['label'] != 0]

#set X & Y training from df for ML models
X = df.drop(labels=['label'], axis=1)
X = np.asarray(X) #convert to array

Y = df['label']
Y = np.asarray(Y) #convert to array
Y = Y.astype(int)

Y = Y-1 #must start at 0 for lightgbm to work properly
'''
print(np.unique(Y_train)) #verify that there are 3 classes: 0, 1, 2
(unique, counts) = np.unique(Y_train, return_counts=True)
freq = np.asarray((unique, counts)).T
print(freq)
'''

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=20)

############################################################
#################     lightgbm        ######################
############################################################
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

lgb_engine=lgb.LGBMClassifier()

#create dataset for lgb
lgb_data = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_test, label=y_test)

### random tune lgb model ### to get the initial best parameters dialed in 
random_params = {
    'learning_rate': np.linspace(.1, 1, 9),
    'num_trees':range(1, 10, 1),
    'num_leaves':range(10,80,5),
    'max_depth':range(2,10,1),
    'min_sum_hessian_in_leaf':range(1,10,1),
    'num_threads':[8],
    'objective':['multiclass'],
    'num_class':[3],
    'metric':['multi_logloss']      
    }

#setup random tuning with parameters above
lgb_random = RandomizedSearchCV(lgb_engine, random_params, verbose=1,
                                cv=5, random_state=42) 

#run random tuning on lgb
rand_tune = lgb_random.fit(X_train, y_train)

rand_tune.best_params_
'''
{'objective': 'multiclass',
 'num_trees': 9,
 'num_threads': 8,
 'num_leaves': 25,
 'num_class': 3,
 'min_sum_hessian_in_leaf': 2,
 'metric': 'multi_logloss',
 'max_depth': 7,
 'learning_rate': 0.775}
'''
lgb_params = rand_tune.best_params_

lgb_model = lgb.LGBMClassifier()
lgb_model.set_params(**lgb_params)
lgb_model = lgb_model.fit(X_train, y_train)

#save model
#lgb_mod_name = git_dir+'lgb_mod.sav'
#pickle.dump(lgb_model, open(lgb_mod_name, 'wb'))


###################################################################
#################   Model Prediction & Accuracy   #################
###################################################################
from keras.metrics import MeanIoU
from sklearn import metrics

#predict test set
#mask_only = np.where(y_test < 5) #this strips out background to match model of 0,1,2
#values_only_mask = y_test[mask_only]
#values_only_preds = x_test[mask_only]

preds = lgb_model.predict(X_test)

print('Lightgbm Performance - Labeled Pixels Only')
print('Accuracy = ', metrics.accuracy_score(
        y_test, preds))

#calculate IoU accuracy for each class
num_classes=3
IOU_keras = MeanIoU(num_classes=num_classes)  
IOU_keras.update_state(y_test, preds)

print("Mean IoU =", IOU_keras.result().numpy())
values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] +
               values[1,0]+ values[2,0])
class2_IoU = values[1,1]/(values[1,0] + values[1,1] + values[1,2] +
               + values[0,1]+ values[2,1])
class3_IoU = values[2,2]/(values[2,0] + values[2,1] + values[2,2] +
               values[0,2]+ values[1,2])

#print IoU for each pixel
print('Class 1 (wood pellet): ' + str(class1_IoU))
print('Class 2 (ldpe): ' + str(class2_IoU))
print('Class 3 (other): ' + str(class3_IoU))

###################################################################
#################      Predict on New Images      #################
###################################################################
from PIL import Image
from numpy import asarray

#####################################
######  Prep Images to Segment ######
#####################################

#input folder path where images are stored
new_images_in = my_dir+'new_images_in/' #must have / at the end

#output folder path
pred_images_out = my_dir+'output_images_gabor' #folder to save output images

#crop image dimensions
left = 0
top = 165
right = 2262
bottom = 530

#get all files names in folder to iterate through
new_in_folder = os.listdir(new_images_in)

#set shape of new images to revert back to after processing
img_shape = Image.open(new_images_in+new_in_folder[0])
img_shape = img_shape.crop((left, top, right, bottom))
img_shape = asarray(img_shape)
img_shape = img_shape.shape[-3:-1]

#####################################
######  Segment and Save Images #####
#####################################

#load, process, predict, reshape and save each image
for img in new_in_folder:
    image = Image.open(new_images_in+img) #read in 1 image at a time
    cropped = image.crop((left, top, right, bottom)) #crop image
    img_array = asarray(cropped) #convert to array
    new_img = np.expand_dims(img_array, axis=0) #shape for VGG16
    new_img = new_model.predict(new_img) #get VGG16 weights
    new_features = new_img.reshape(-1, new_img.shape[3]) #shape
    
    #predict with chosen model (above -- uncomment model) here
    new_preds = lgb_model.predict(new_features)
    
    #shape into image file
    new_pred_img = new_preds.reshape(img_shape)
    
    #save image to chosen folder (above)
    plt.imsave(pred_images_out+img, new_pred_img, cmap='gray')


