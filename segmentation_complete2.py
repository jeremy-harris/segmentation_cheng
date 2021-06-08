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
new_model.save('./segmentation_cheng/new_model.h5')

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

df = pd.DataFrame(X)
df['Label'] = Y

#Look to verify labels are correct (i.e. not all blank)
#print(df['Label'].unique()) #get different label values (pixel values)
#print(df['Label'].value_counts()) #get sum of each pixel value

#Drop pixels with value of 0 -- no need to detect the background in this case
df2 = df[df['Label'] != 0]

#set X & y training from df for ML models
X_train = df2.drop(labels=['Label'], axis=1)
X_train = np.asarray(X_train) #convert to array
#X_train = X_train.values 
Y_train = df2['Label']
Y_train = np.asarray(Y_train) #convert to array
#Y_train = Y_train.values #convert to arrary


###### testing features ######
#get test files brought in to match dimensions
test_images = get_files(my_dir, test_img_path, 1)
test_masks = get_files(my_dir, test_mask_path, 0)

#send test images through VGG16 model to get weights (one at a time)
test_img_pred = get_features(test_images)

i = 0

while i < len(test_images):
    img = np.expand_dims(test_images[i], axis=0)
    get_features = new_model.predict(img)
    get_features = np.squeeze(get_features)
    test_img_pred.append(get_features)
    
    i += 1

test_img_pred = np.array(test_img_pred)
#now make match our size for train features
test_features = test_img_pred.reshape(-1, test_img_pred.shape[3])

#load test mask
test_masks = []
for m in sorted(glob.glob(os.path.join(test_mask_path, "*.tif"))):
    test_mask_in = cv2.imread(m, 0)
    test_masks.append(test_mask_in)
    
test_masks = np.array(test_masks)
#shape to match size of test images
test_masks = test_masks.reshape(-1)

##############################################################
#################     Pre-Processing     #####################
##############################################################

### Normalize Data ###
#normalize our data to a scale from 0 to 1 for ML improvements. We only
#need to do this for the X values (test and train) not the Y values as those
#are our target pixel values 0-4
from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train) #x train values
test_norm = preprocessing.normalize(test_features) #x test values

### PCA ###
#PCA for reduction of the size of data (reduce 64 features to 25)
from sklearn.decomposition import PCA
pca = PCA(.99) #setup PCA so that it will retain 99% of the variance
pca.fit(X_train_norm) #get PCAs of the training images/features
pca.n_components_ #this shows that we only need 25 components to achieve 99%

#save off pca for later use
with open('pca.pkl', 'wb') as pickle_file:
    pickle.dump(pca, pickle_file)

#apply pca to transform training and testing images
X_train_pca = pca.transform(X_train_norm)
test_pca = pca.transform(test_norm)

##############################################################
#################     Random Forest     ######################
##############################################################
'''
#This has already been completed. Models were saved as RF_base.sav & 
#tuned model is RF_tuned.sav.

import random
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import RandomizedSearchCV

#Base RF model to get it working for comparison
rf_base = RF(n_estimators=50,
                    random_state = 42,
                    n_jobs=-1)

### Train Base RF Model ###
t1 = time.time()
random.seed(30)
rf_base.fit(X_train_pca, Y_train)
t2 = time.time()
print("Time in Seconds to Train: " + str(round(t2-t1))) #43 seconds on 20 cores

#save base RF model for later use
RF_base_name = 'RF_base.sav'
pickle.dump(rf_base, open(RF_base_name, 'wb'))

#################### RANDOM GRID SEARCH TUNING ##################
#Find opitmal tuning parameters for RF
###    NOTE: This will take several hours  ###

#Now use random search to try various combinations
rf = RF() #set base model of RF

#setup random grid parameter ranges
random_grid = {'n_estimators': range(20, 500, 20),
               'max_features': ['auto'],
               'max_depth': range(10,100, 10),
               'min_samples_split': range(2,50,10),
               'min_samples_leaf': range(1,10,10),
               'bootstrap': [True]}

# Random search of parameters, using 3 fold cross validation, 
# search across 200 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 200, #try 200 dif. random combos
                               cv = 3,
                               verbose=2,
                               random_state=42,
                               n_jobs = -2) #all but 1 processor

##### TUNE WITH GRID SEARCH ###
#tune with random grid
t1 = time.time()

random.seed(30)
rf_tuned = rf_random.fit(X_train_pca, Y_train)
t2 = time.time()

print("Time in Seconds to Tune: " + str(round(t2-t1)))
#53,258 (14.79 hours) on a system using 39 cores

#print best parameters
print(rf_random.best_params_)
#'n_estimators': [440]
#'max_features': ['auto'],
#'max_depth': [60]
#'min_samples_split':[2]
#'min_samples_leaf': [1]
#'bootstrap': [True]}

#save base RF model for later use
RF_tune_name = 'RF_tuned.sav'
pickle.dump(rf_tuned, open(RF_tune_name, 'wb'))


### Manually set model -- Looking to see why it is so big when saved
#print best parameters
print(RF_tuned_model.best_params_)
rf_params = RF_tuned_model.best_params_

#create XRFmodel with params
rf2_model = RF(**rf_params)

t1 = time.time()

random.seed(30)
rf2_model_fit = rf2_model.fit(X_train_pca, Y_train)

t2=time.time()
print("Time in Minutes to Tune: " + str(round((t2-t1)/60)))
# Time to tune on 40 core system: 104 min

#save model
RF_tune2_name = 'RF_tune2.sav'
pickle.dump(rf2_model_fit, open(RF_tune2_name, 'wb'))

'''
############################################################
#################     XG Boosting     ######################
############################################################
import random
from xgboost import XGBClassifier as XGB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

### XGB Base ###
#set base parameters
xg_params = {'max_depth':8, 'eta':.5, 'objective':'multi:softmax',
             'eval_metric':'mlogloss', 'num_class':3, 'tree_method':'hist',
             'gamma':0, 'n_estimators':100, 'subsample':0.9}
num_round = 1000

#train model with cpu --- no pca, no norm
xgb_t1 = time.time()
xgb_model = XGB()
xgb_model.set_params(**xg_params)
xgb_base = xgb_model.fit(X_train, Y_train) #non-pca
xgb_t2 = time.time()
print("Base XGB Elapsed time in seconds: ", round(xgb_t2 - xgb_t1))
#####################################################
# 40x CPU - model time with params above was 14 seconds  #

#save model
xg_base_name = 'XG_base_3NoPCA.sav'
pickle.dump(xgb_base, open(xg_base_name, 'wb'))

############
#train model with cpu --- NORM, no pca
xgb_t1 = time.time()
xgb_model = XGB()
xgb_model.set_params(**xg_params)
xgb_baseNORM = xgb_model.fit(X_train_norm, Y_train) #NORM non-pca
xgb_t2 = time.time()
print("Base XGB Elapsed time in seconds: ", round(xgb_t2 - xgb_t1))
#####################################################
# 40x CPU - model time with params above was 15 seconds  #

#save model
xg_baseNORM_name = 'XG_base_3Norm.sav'
pickle.dump(xgb_baseNORM, open(xg_baseNORM_name, 'wb'))

#############
#train model with cpu --- NORM & PCA
xgb_t1 = time.time()
xgb_model = XGB()
xgb_model.set_params(**xg_params)
xgb_basePCA = xgb_model.fit(X_train_pca, Y_train) 
xgb_t2 = time.time()
print("Base XGB Elapsed time in seconds: ", round(xgb_t2 - xgb_t1))
#####################################################
# 40x CPU - model time with params above was 13 seconds  #

#save model
xg_basePCA_name = 'XG_base_3PCA.sav'
pickle.dump(xgb_basePCA, open(xg_basePCA_name, 'wb'))





#####  Random Tuning CPU #####
##############################
'''
#Attempt to change output location to keep from fillng up HD for gpu usage
#import os
#os.environ['JOBLIB_TEMP_FOLDER'] = '/mnt/MLshare'

#set parameters for gridsearch tuning
xg_tune_params = {'max_depth':range(3,30,10),
                  'eta': np.linspace(0.2,1,10),
                  'objective':['multi:softmax'],
                  'min_child_weight':range(1,10,10),
                  'subsample':np.linspace(.1, 1, 10),
                  'colsample_bytree':np.linspace(.1,1,10),
                  'eval_metric':['mlogloss'],
                  'num_class':[4],
                  'tree_method':['hist'],
                  'seed':[30]}

# Random search of parameters, using 3 fold cross validation, 
# search across 200 different combinations, and use all available cores
xg = XGB()
xg_random2 = RandomizedSearchCV(estimator = xg,
                               param_distributions = xg_tune_params2,
                               n_iter = 50, #try 200 dif. random combos
                               cv = 3,
                               verbose=2,
                               random_state=42,
                               n_jobs = -1) #all but 1 processors

### Grid Search ###
#Dial in from the Random Grid search tuning
xg = XGB()
xg_tune_params2 = {'max_depth':[7, 8, 9],
                  'eta': np.linspace(0.1, 0.3, 5),
                  'objective':['multi:softmax'],
                  'min_child_weight':[1],
                  'subsample':np.linspace(0.7, 0.9, 5),
                  'colsample_bytree':[1],
                  'eval_metric':['mlogloss'],
                  'num_class':[3],
                  'tree_method':['hist'],
                  #'predictor':['gpu_predictor'], #use gpu to predict
                  'gpu_id':[0],
                  'seed':[30]}

#assign model with parameters
xg_tuned2 = GridSearchCV(xg, xg_tune_params2,
                         cv = 3, n_jobs = -1, verbose=2)

#Find best model using only 3 classes
t1 = time.time()

random.seed(30)
xg_tuned = xg_tuned2.fit(X_train, Y_train) #no pca or normalization
t2 = time.time()

print("Time in Minutes to Tune: " + str(round((t2-t1)/60)))
#40 core workstation: 119 minutes for above random grid


##### TUNE WITH GRID SEARCH ###
#tune with random grid
t1 = time.time()

random.seed(30)
xg_tuned = xg_random2.fit(X_train_pca, Y_train)
t2 = time.time()

print("Time in Minutes to Tune: " + str(round((t2-t1)/60)))
#40 core workstation: 119 minutes for above random grid

#print best parameters
print(xg_random2.best_params_)

##### Hard Set Parameters from XGB Model ######
xg_best_params = xg_random2.best_params_
xg_best_params['predictor'] = 'gpu_predictor' #allow gpu for predictions
xg_best_params['gpu_id'] = 0     #assign gpu0 as the default gpu

#create XGB model with params
xg_gpu_mod = XGB(xg_best_params)

t1 = time.time()

random.seed(30)
xg_best_gpu = xg_gpu_mod.fit(X_train_pca, Y_train)

t2=time.time()
print("Time in Minutes to Tune: " + str(round((t2-t1)/60)))
# Time to tune on 40 core system:   2 minutes

#save model
xg_tuned_gpu_name = 'XG_gpu_tuned.sav'
pickle.dump(xg_tuned, open(xg_tuned_gpu_name, 'wb'))

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

