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

# location of files on WATT
# =============================================================================
# my_dir = "/root/mlhome/segmentation/segmentation_cheng/"
# git_dir = "/root/mlhome/segmentation/segmentation_cheng/"
# img_path = "./train_images/"
# mask_path = "./train_masks/"
# =============================================================================

my_dir = "/home/jeremy/Projects/segmentation_cheng/"
git_dir = "/home/jeremy/Projects/segmentation_cheng/"
img_path = "./train_images/"
mask_path = "./train_masks/"


def get_files(c_dir, path, cv2_flag): #flag 1 = color, 0 = grayscale
    os.chdir(c_dir)
    files = []
    for i in sorted(glob.glob(os.path.join(path, "*.*"))):
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

'''
### tone to get gabor features ###
def get_gabor(input_array):
    i=0
    df_out = pd.DataFrame()
    while i < len(input_array):
        img = input_array[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1)
        df = pd.DataFrame()
        df['orig'] = img
        num = 1  #To count numbers up in order to give Gabor features a label in the data frame
        kernels = []

        for theta in (0, np.pi*.125, np.pi*.25, np.pi*.375, np.pi*.5, np.pi*.675, np.pi*.75, np.pi*.875): #orientation on 360 degree 
            for sigma in (1, 5, 10):  #size of the envelope and lines
                for lamda in (3, 5, 10):   #length of the wavelength
                    for gamma in (.05, .1):   #height of the gabor feature
                        for ksize in (5, 25, 45):
                            
                            gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                            kernels.append(kernel)
                            #Now filter the image and add values to a new column 
                            fil_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                            fil_img = fil_img.reshape(-1)
                            df[gabor_label] = fil_img  #Labels columns as Gabor1, Gabor2, etc.
                            print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma, ': ksize=', ksize)
                            num += 1  #Increment for gabor column label
        print(i, " of ", len(input_array))

        
        #img_label = 'img_'+str(i)
        df_out = df_out.append(df, ignore_index=True)
               
        i += 1
        
    return(df_out, kernels)

X2, kernels = get_gabor(train_images)

#add VGG features and gabor features together
df = pd.concat([X1, X2], axis=1)

#reshape Y to match X
Y = train_masks.reshape(-1)

df['label'] = Y

#drop features that don't record a value (i.e. sum of values = 0)
df.drop([col for col, val in df.sum().iteritems() if val ==0], axis=1, inplace=True)

#Drop pixels with value of 0 -- no need to detect the background in this case
df = df[df['label'] != 0]

#set X & Y training from df for ML models
X = df.drop(labels=['label'], axis=1)
'''
##### Build Specific features based on model tuning/performance ######
gabor_data = [
    ['g_01', 25, 5, 2.12, 10, .1, 0, 'cv2.CV_32F'],
    ['g_02', 45, 10, 0.39, 10, .1, 0, 'cv2.CV_32F'],
    ['g_03', 45, 5, 2.12, 10, .1, 0, 'cv2.CV_32F'],
    ['g_04', 45, 5, 0.785, 3, .05, 0, 'cv2.CV_32F'],
    ['g_05', 45, 5, 1.178, 10, .1, 0, 'cv2.CV_32F'],
    ['g_06', 45, 5, 0.39, 3, .05, 0, 'cv2.CV_32F'],
    ['g_07', 45, 10, 0.785, 3, .05, 0, 'cv2.CV_32F'],
    ['g_08', 45, 10, 0.39, 3, .05, 0, 'cv2.CV_32F'],
    ['g_09', 5, 10, 1.178, 5, .05, 0, 'cv2.CV_32F'],
    ['g_10', 45, 5, 0.39, 3, .1, 0, 'cv2.CV_32F'],
    ['g_11', 45, 10, 0.39, 10, .05, 0, 'cv2.CV_32F'],
    ['g_12', 45, 10, 0.785, 3, .1, 0, 'cv2.CV_32F'],
    ['g_13', 5, 5, 0.0, 5, .05, 0, 'cv2.CV_32F'],
    ['g_14', 45, 10, 0.0, 5, .05, 0, 'cv2.CV_32F'],
    ['g_15', 45, 10, 0.0, 5, .1, 0, 'cv2.CV_32F'],
    ['g_16', 45, 10, 0.785, 5, .1, 0, 'cv2.CV_32F'],
    ['g_17', 25, 5, 2.12, 10, .05, 0, 'cv2.CV_32F'],
    ['g_18', 45, 10, 0.0, 3, .1, 0, 'cv2.CV_32F'],
    ['g_19', 5, 5, 0.785, 3, .05, 0, 'cv2.CV_32F'],
    ['g_20', 45, 10, 1.178, 10, .1, 0, 'cv2.CV_32F'],
    ['g_21', 45, 5, 1.78, 10, .05, 0, 'cv2.CV_32F'],
    ['g_22', 25, 5, 0.785, 3, .05, 0, 'cv2.CV_32F'],
    ['g_23', 45, 5, 1.178, 5, .05, 0, 'cv2.CV_32F'],
    ['g_24', 25, 5, 2.12, 5, .05, 0, 'cv2.CV_32F'],
    ['g_25', 25, 5, 0.0, 3, .05, 0, 'cv2.CV_32F'],
    ['g_26', 45, 5, 2.12, 10, .05, 0, 'cv2.CV_32F'],
    ['g_27', 25, 10, 0.39, 3, .1, 0, 'cv2.CV_32F'],
    ['g_28', 45, 5, 0.785, 5, .05, 0, 'cv2.CV_32F']
    ]
g_df = pd.DataFrame(gabor_data, columns =['num', 'ksize', 'sigma', 'theta', 'lamda', 'gamma', 'psi', 'ktype'])
g_ar = np.asarray(g_df)

def get_tuned_gabors(input_array):
    i=0
    df_out = pd.DataFrame()
    while i < len(input_array):
        img = input_array[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1)
        df = pd.DataFrame()
        df['orig'] = img
        num = 1  #To count numbers up in order to give Gabor features a label in the data frame
        kernels = []

        print("")        
        print(i+1, " of ", len(input_array))
        for g in range(0, len(g_ar)):
            ksize = g_ar[g,1]
            sigma = g_ar[g,2]
            theta = g_ar[g,3]
            lamda = g_ar[g,4]
            gamma = g_ar[g,5]
            
            gabor_label = 'Gabor' + str(num)
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
            kernels.append(kern)
            
            #Now filter the image and add values to a new column 
            fil_img = cv2.filter2D(img, cv2.CV_8UC3, kern)
            fil_img = fil_img.reshape(-1)
            df[gabor_label] = fil_img  #Labels columns as Gabor1, Gabor2, etc.
            print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma, ': ksize=', ksize)
            num += 1  #Increment for gabor column label
        
        

        
        #img_label = 'img_'+str(i)
        df_out = df_out.append(df, ignore_index=True)
               
        i += 1
        
    return(df_out, kernels)

X_gab, kerns2 = get_tuned_gabors(train_images)

########## get other filters ##########
#CANNY EDGE
from skimage.filters import roberts, sobel, scharr
from scipy import ndimage as nd
from tqdm import tqdm

def get_filters(input_array):
    i=0
    df_out = pd.DataFrame()
    for i in tqdm(range(0, len(input_array)), desc="Getting Filters"):
        img = input_array[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        df = pd.DataFrame()
        filters = []
    
        #canny edge    
        edges = cv2.Canny(img, 100,200)   #Image, min and max values
        edges1 = edges.reshape(-1)
        df['canny'] = edges1 #Add column to original dataframe
        filters.append(edges)

        #ROBERTS EDGE
        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        df['roberts'] = edge_roberts1
        filters.append(edge_roberts)

        #SOBEL
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['sobel'] = edge_sobel1
        filters.append(edge_sobel)

        #SCHARR
        edge_scharr = scharr(img)
        edge_scharr1 = edge_scharr.reshape(-1)
        df['scharr'] = edge_scharr1
        filters.append(edge_scharr)
        
        #GAUSSIAN with sigma=3
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)
        df['gaussianS3'] = gaussian_img1
        filters.append(gaussian_img)
        
        #GAUSSIAN with sigma=7
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img3 = gaussian_img2.reshape(-1)
        df['gaussianS7'] = gaussian_img3
        filters.append(gaussian_img2)
        
        #MEDIAN with sigma=3
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)
        df['median'] = median_img1
        filters.append(median_img)
        
        df_out = df_out.append(df, ignore_index=True)
               
        i += 1
        
    return(df_out, filters)

X_fil, filters = get_filters(train_images)


########################################


#add VGG features and gabor features together
df = pd.concat([X1, X_gab, X_fil], axis=1)

#reshape Y to match X
Y = train_masks.reshape(-1)

df['label'] = Y

#drop features that don't record a value (i.e. sum of values = 0)
#df.drop([col for col, val in df.sum().iteritems() if val ==0], axis=1, inplace=True)

#Drop pixels with value of 0 -- no need to detect the background in this case
df = df[df['label'] != 0]

#set X & Y training from df for ML models
X = df.drop(labels=['label'], axis=1)

#setup Y for lightgbm models
Y = df['label']
Y = Y.astype(int)

Y = Y-1 #must start at 0 for lightgbm to work properly

'''
### PCA ###
#PCA for reduction of the size of data
from sklearn.decomposition import PCA
pca = PCA(.99) #setup PCA so that it will retain 99.9% of the variance
pca.fit(X) #get PCAs of the training images/features
pca.n_components_ 

#save off pca for later use
#with open('pca.pkl', 'wb') as pickle_file:
#    pickle.dump(pca, pickle_file)

#apply pca to transform training and testing images
X_pca = pca.transform(X)

#setup Y for lightgbm models
Y = df['label']
Y = Y.astype(int)

Y = Y-1 #must start at 0 for lightgbm to work properly
'''
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
'''
base vgg16 results (64 features)
Class 1 (wood pellet): 0.68524706
Class 2 (ldpe): 0.95273745
Class 3 (other): 0.9986113
'''

'''
feature importance
XX = pd.DataFrame(X)
feature_list = list(XX.columns)
feature_imp = pd.Series(lgb_model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp[0:50])
#gabor 90, 51, 21, 2
#2:  theta=0, sigma=10, lambda=0.1, gamma =0.05, ksize=10
#21: theta= 0.0 : sigma= 10 : lamda= 1.6707963267948966 : gamma= 0.5 : ksize= 10
#51: theta= 0.0 : sigma= 30 : lamda= 1.6707963267948966 : gamma= 0.05 : ksize= 50
#90: theta= 0.7853981633974483 : sigma= 10 : lamda= 2.456194490192345 : gamma= 0.05 : ksize= 25
'''

###################################################################
#################      Predict on New Images      #################
###################################################################
from PIL import Image
from numpy import asarray

#set path to images
new_img_path = "new_images_in/"
pred_output = "output_images_gabor/"

#load, process and predict on new images
new_images = get_files(my_dir, new_img_path, 1)

#crop images to match model dimension images
#crop image dimensions
left = 0
top = 165
right = 2262
bottom = 530

#get all files names in folder to iterate through
new_in_folder = os.listdir(my_dir+new_img_path)

###### crop new images to match model ######
new_cropped = []
for img in new_in_folder:
    image = Image.open(my_dir+new_img_path+img) #read in 1 image at a time
    cropped = image.crop((left, top, right, bottom)) #crop image
    img_array = asarray(cropped) #convert to array
    new_img = np.expand_dims(img_array, axis=0) #shape for VGG16
    new_img = np.squeeze(new_img)
    new_cropped.append(new_img)
new_cropped = np.array(new_cropped)
cropped_shape = new_cropped.shape[1:3] #get shape of cropped image for later use

###### get features one by one and output prediction #######  

i=0
for img in new_cropped:
    #get name
    img_name = new_in_folder[i]
    #get vgg16 features
    img = new_cropped[i]
    img_in = np.expand_dims(img, axis=0)
    X1_new = get_features(img_in)
    X1_new = pd.DataFrame(X1_new)

    #get gabor features
    X2_new, kerns_new = get_tuned_gabors(img_in)
    X2_new = pd.DataFrame(X2_new)
    
    #get filter features
    X_fil, filters = get_filters(img_in)
    
    #add VGG features and gabor features together
    df2 = pd.concat([X1_new, X2_new, X_fil], axis=1)
    
    #drop na values or ones that sum to 0
    #df2.drop([col for col, val in df2.sum().iteritems() if val ==0], axis=1, inplace=True)
    
    #apply pca
    #X_pca = pca.transform(df2)
    #df_pca = pd.DataFrame(X_pca)
    
    #predict on image
    new_pred = lgb_model.predict(df2)
    
    #shape into original image shape
    new_pred_img = new_pred.reshape(cropped_shape)
    
    #save image to chosen folder (above)
    plt.imsave(my_dir+pred_output+img_name, new_pred_img, cmap='gray')
    i += 1

# review filters for tuning
'''
kerns= []
n=1
for k in (0, np.pi*.125, np.pi*.25, np.pi*.375, np.pi*.5, np.pi*.675, np.pi*.75, np.pi*.875):
    ksize=25 #pixel size
    sigma=5 #size of envelope
    theta=k #orienation of gabor
    lamda=5 #width of stripe
    gamma=0.1 #height of the gabor
    

    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
    print(n, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma, ': ksize=', ksize)
    n+=1
    kerns.append(kern)

ss=10
fig = plt.figure(figsize=(ss, ss))
columns = 3
rows = 3

z=0
for i in range(1,rows*columns+1):
    img = kerns[z]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(z+1)
    z+=1
    

plt.show()    
'''  
    
    
    
    
