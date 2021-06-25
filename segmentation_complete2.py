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
import time


############################################################
#################     Load Images     ######################
############################################################

#set path to images; I had 26 images so I split them 70/30 and put the 
#image into the folder locations below. 18 training, 8 testing.

my_dir = "/root/mlhome/segmentation"
git_dir = "/root/mlhome/segmentation/segmentation_cheng/"
train_path = "./cropped_train_frames/"
mask_path = "./cropped_train_masks/"
test_img_path = "./cropped_test_frames/"
test_mask_path = "./cropped_test_masks/"


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

train_images = get_files(my_dir, train_path, 1)
train_masks = get_files(my_dir, mask_path, 0)

#set x & y up for standard ML 
X_train = train_images
y_train = train_masks

##########################################################
#################     VGG16 Work    ######################
##############################################array############

#Pull the VGG16 model and extract the weights from VGG16 training on imagenet
#include_top=False removes the dense layers for prediction

#set image size, this is useful if images aren't of the same size
IMG_W = X_train.shape[2]
IMG_H = X_train.shape[1]

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

#because of the large image size, the system will throw OOM (out of memory)
#errors if trying to run all images through the new_model at once. To work
#around this, I created loop to pull an image one at a time and then append
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

#### training features ####
X = get_features(X_train)

#reshape Y to match X
Y = y_train.reshape(-1)

#Here we want to drop pixels with a value of 0 as it is unlabeled and
#not worth wasting resources on to detect.

def make_df(x, y):
    df = pd.DataFrame(x)
    df['Label'] = y

    #Look to verify labels are correct (i.e. not all blank)
    #print(df['Label'].unique()) #get different label values (pixel values)
    #print(df['Label'].value_counts()) #get sum of each pixel value

    #Drop pixels with value of 0 -- no need to detect the background in this case
    df = df[df['Label'] != 0]

    #set X & y training from df for ML models
    X = df.drop(labels=['Label'], axis=1)
    X = np.asarray(X) #convert to array

    Y = df['Label']
    Y = np.asarray(Y) #convert to array
    Y = Y.astype(int)
    return(X, Y)

X_train, Y_train = make_df(X, Y)
Y_train = Y_train-1
'''
print(np.unique(Y_train)) #verify that there are 3 classes: 1, 2, 3
(unique, counts) = np.unique(Y_train, return_counts=True)
freq = np.asarray((unique, counts)).T
print(freq)
'''

###### testing features ######
#get test files brought in to match dimensions
test_images = get_files(my_dir, test_img_path, 1)
test_masks = get_files(my_dir, test_mask_path, 0)

#send test images through VGG16 model to get weights (one at a time)
x = get_features(test_images)
y = test_masks.reshape(-1)

x_test, y_test = make_df(x, y)
y_test = y_test-1
#print(np.unique(y_test)) #verify that there are 3 classes: 1,2,3

##############################################################
#################     Pre-Processing     #####################
##############################################################

### Normalize Data ###
#normalize our data to a scale from 0 to 1 for ML improvements. We only
#need to do this for the X values (test and train) not the Y values as those
#are our target pixel values 0-4
from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train) #x train values
x_test_norm = preprocessing.normalize(x_test) #x test values

### PCA ###
#PCA for reduction of the size of data (reduce 64 features to 25)
from sklearn.decomposition import PCA

pca_train = PCA(.99) #setup PCA so that it will retain 99% of the variance
pca_train.fit(X_train_norm) #get PCAs of the training images/features
#pca_train.n_components_ #this shows that we only need 25 components to achieve 99%

pca_test = PCA(.99) #setup PCA so that it will retain 99% of the variance
pca_test.fit(x_test_norm) #get PCAs of the training images/features
#pca_test.n_components_ #this shows that we only need 25 components to achieve 99%

#apply pca to transform training and testing images
X_train_pca = pca_train.transform(X_train_norm)
x_test_pca = pca_test.transform(x_test_norm)

#save off pca's for later use

'''
with open(git_dir+'pca_train.pkl', 'wb') as pickle_file:
    pickle.dump(pca_train, pickle_file)
    
with open(git_dir+'pca_test.pkl', 'wb') as pickle_file:
    pickle.dump(pca_test, pickle_file)
'''

############################################################
#################     lightgbm        ######################
############################################################
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import random

lgb_engine=lgb.LGBMClassifier()

#create dataset for lgb
lgb_data = lgb.Dataset(X_train_pca, label=Y_train)
lgb_val = lgb.Dataset(x_test_pca, label=y_test)

### random tune lgb model ### to get the initial best parameters dialed in 
random_params = {
    'learning_rate': np.linspace(.1, 1, 10),
    'num_trees':range(1, 10, 1),
    'num_leaves':range(10,80,5),
    'max_depth':range(2,10,1),
    'min_sum_hessian_in_leaf':range(1,10,1),
    'num_threads':[8],
    'objective':['multiclass'],
    'num_class':[3]          
    }

#setup random tuning with parameters above
lgb_random = RandomizedSearchCV(lgb_engine, random_params, verbose=1,
                                cv=5, random_state=42) 

#run random tuning on lgb
rand_tune = lgb_random.fit(X_train_pca, Y_train)

rand_tune.best_params_
'''
{'objective': 'multiclass',
 'num_trees': 7,
 'num_threads': 8,
 'num_leaves': 75,
 'num_class': 3,
 'min_sum_hessian_in_leaf': 1,
 'min_data_in_leaf': 5,
 'max_depth': 7,
 'learning_rate': 0.2}
'''
best_params = rand_tune.best_params_

gbm = lgb.train(best_params, lgb_data, num_boost_round=20, valid_sets=lgb_val, early_stopping_rounds=5)


#### predict and get accuracy
y_pred = rand_tune.predict(x_test_pca)
accuracy = accuracy_score(y_pred, y_test)
print('Lightgbm accuracy: {0:0.4f}'.format(accuracy))

#save model
lgb_mod_name = 'lgb_mod.sav'
pickle.dump(git_dir+rand_tune, open(lgb_mod_name, 'wb'))

'''
###################################################################
#################   Model Prediction & Accuracy   #################
###################################################################

# This section loads the models of interest that we have already trained
# and saved, then defines a function which takes in 4 arguments (model, 
# test features, test labels and number of predicted classes) and then 
# computes the prediction of the testing data and computes the accuracy
# of the model to predict ONLY the areas of the image that correspond to 
# the areas of the mask that have been labeled. Further, the model then
# computes the ability to successfully predict each pixel independently.

from sklearn.ensemble import RandomForestClassifier as RF

#Load all trained models models to compare
RF_base_name = 'RF_base.sav'
RF_tuned_name = 'RF_tuned.sav'
RF_tuned2_name = 'RF_tune2.sav'
XG_base_name = 'XG_base_3NoPCA.sav'
XG_base_NORM_name = 'XG_base_3Norm.sav'
XG_base_PCA_name = 'XG_base_3PCA.sav'
XG_tuned_name = 'XG_tuned.sav'
XG_tuned_gpu_name = 'XG_gpu_tuned.sav'

RF_base_model = pickle.load(open(RF_base_name, 'rb'))
RF_tuned_model = pickle.load(open(RF_tuned_name, 'rb'))
RF_tuned2_model = pickle.load(open(RF_tuned2_name, 'rb'))
XG_baseNORM_model = pickle.load(open(XG_base_NORM_name, 'rb'))
XG_basePCA_model = pickle.load(open(XG_base_PCA_name, 'rb'))
XG_base_model = pickle.load(open(XG_base_name, 'rb'))


XG_tuned_model = pickle.load(open(XG_tuned_name, 'rb'))
XG_tuned_GPU_model = pickle.load(open(XG_tuned_gpu_name, 'rb'))

#assign best params to tuned models
RF_tuned_best = RF_tuned_model.best_estimator_
XG_tuned_best = XG_tuned_model.best_estimator_
XG_tuned_GPU = XG_tuned_GPU_model.best_estimator_

####################### START FUNCTION ################################
###function for model evaluation ###
# accuracy is only computed for the sections of images that match the 
# labeled sections of the masks. We compare how well our model predicts
# the matching areas of the image to the labeled area of the masks
from sklearn import metrics
from keras.metrics import MeanIoU

#create pipeline for models that will provide accuracy of each
def evaluate(model, test_features, test_labels, num_classes):
    t1 = time.time()
    global mask_values_only, preds_values_only, predictions, values
    predictions = model.predict(test_features)
    
    #get labeled pixels only by subsetting mask for pixels that have been
    #labeled and then applying that subset of value to mask & prediction
    mask_only = np.where(test_labels > 0)
    mask_values_only = test_labels[mask_only]
    preds_values_only = predictions[mask_only]
    
    #print results for accuracy of labeled pixels
    print('Model Performance')
    print('Accuracy = ', metrics.accuracy_score(
            mask_values_only, preds_values_only))
    
    #calculate accuracy for each pixel prediction
    IOU_keras = MeanIoU(num_classes=num_classes)  
    IOU_keras.update_state(mask_values_only, preds_values_only)
    print("Mean IoU =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
    class1_IoU = values[1,1]/(values[1,1] + values[1,2] + values[1,3] +
                   values[2,1]+ values[3,1])
    class2_IoU = values[2,2]/(values[2,1] + values[2,2] + values[2,3] +
                   + values[1,2]+ values[3,2])
    class3_IoU = values[3,3]/(values[3,1] + values[3,2] + values[3,3] +
                   values[1,3]+ values[2,3])

    #print IoU for each pixel
    print('Class 1 (wood pellet): ' + str(class1_IoU))
    print('Class 2 (ldpe): ' + str(class2_IoU))
    print('Class 3 (other): ' + str(class3_IoU))
    
    t2 = time.time()
    print("Elapsed Training time in seconds: ", t2 - t1)
######################### END FUNCTION ################################

#####   Get Model Accuracy & Compare   #####
#get model accuracy for all models on test images with function above
num_classes = 4

### BASE RF ###
#base model: 96.274% // 55.169% Pixel 1
preds_rf_base = evaluate(RF_base_model, test_pca, test_masks, num_classes)

### TUNED RF ###
#best tuned model: 96.402% // 55.995% Pixel 1
preds_rf_tuned = evaluate(RF_tuned_best, test_pca, test_masks, num_classes)

### TUNED RF2 ###
#best tuned model: % // % Pixel 1
t1=time.time()
preds_rf2_tuned = evaluate(RF_tuned2_model, test_pca, test_masks, num_classes)
t2=time.time()
print("Time in minutes: " + str(round(t2-t1)))

####################################
### BASE XG ###    non norm, non pca, 3 classes
#ORIGINAL ACCURACY: base model: 96.405%% // 55.594% Pixel 1

preds_xg_base = evaluate(XG_base_model, test_features, test_masks, num_classes)
#no norm, no pca = 97.59% / 67.96%

preds_xg_Norm = evaluate(XG_baseNORM_model, test_norm, test_masks, num_classes)
#norm, no pca = 97.19% / 63.07%

preds_xg_PCA = evaluate(XG_basePCA_model, test_pca, test_masks, num_classes)
#norm, pca = 96.56% / 57.13%

#####################################

### TUNED XG ###
#best tuned model: 96.660% // 57.322% Pixel 1
preds_xg_tuned = evaluate(XG_tuned_best, test_pca, test_masks, num_classes)

### TUNED XG GPU ### (just faster than regular XG)
#best tuned model: 96.660% // 57.322% Pixel 1
preds_xg_tuned_gpu = evaluate(XG_tuned_GPU, test_pca, test_masks, num_classes)

###################################################################
#################      Predict on New Images      #################
###################################################################
from PIL import Image
from numpy import asarray
import glob
import cv2
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle

#set working directory on your machine:
os.chdir("/root/mlhome/segmentation/")

#####################################
######  Prep Images to Segment ######
#####################################

#model to use
#all options are below, uncomment the one you want to use
#pred_model = RF_tuned_best    #accy: 96.40%  10 img predict time: 17.5 min
#pred_model = XG_tuned_best   #accy:         10 img predict time: 
#pred_model = SVM_base        #accy:         10 img predict time: 

#input folder path where images are stored
new_images_in = './new_images_in/' #must have / at the end

#output folder path
pred_images_out = './out_pred_images/' #folder to save output images

#load pca weights for pre-processing of new images
with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)

#crop image dimensions
left = 0
top = 165
right = 2262
bottom = 530

#set shape of new images to revert back to later
for img in new_imgages_in:
    img_shape = img.shape[-3:-1]

#get all files names in folder to iterate through
files_in_folder = os.listdir(new_images_in)

#set shape of new images to revert back to after processing
img_shape = Image.open(new_images_in+files_in_folder[0])
img_shape = img_shape.crop((left, top, right, bottom))
img_shape = asarray(img_shape)
img_shape = img_shape.shape[-3:-1]

#####################################
######  Segment and Save Images #####
#####################################

#load, process, predict, reshape and save each image
t1=time.time()
for img in files_in_folder:
    image = Image.open(new_images_in+img) #read in 1 image at a time
    cropped = image.crop((left, top, right, bottom)) #crop image
    img_array = asarray(cropped) #convert to array
    new_img = np.expand_dims(img_array, axis=0) #shape for VGG16
    new_img = new_model.predict(new_img) #get VGG16 weights
    new_features = new_img.reshape(-1, new_img.shape[3]) #shape
    new_norm = preprocessing.normalize(new_features) #normalize
    new_pca = pca.transform(new_norm)
    
    #predict with chosen model (above -- uncomment model) here
    new_preds = pred_model.predict(new_pca)
    
    #shape into image file
    new_pred_img = new_preds.reshape(img_shape)
    
    #save image to chosen folder (above)
    plt.imsave(pred_images_out+img, new_pred_img, cmap='gray')
t2=time.time()
print("Time in seconds for 10 images: " + str(round(t2-t1))) 

