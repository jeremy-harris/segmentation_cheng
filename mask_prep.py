#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:15:50 2020

@author: Jeremy Harris
"""

###This script will take dark unit8 masks and brighten them up so that
###we can make sure our training masks match the correct training image

import glob
import cv2
import os

#set mask directory
in_dir = "/root/mlhome/segmentation/PixelLabelData/"
out_dir = "/root/mlhome/segmentation/train_review/" #for human readable
mask_dir = "/root/mlhome/segmentation/train_masks/"
#set dimensions
img_h = 800
img_w = 2560

#convert images from .png to .tif
#create a brightened copy for human readable review
#create tif copy for ml


#get total count of images
images = os.listdir(in_dir)
total_images = len(images)

##save the masks off that have been brightened for "human review" to match
count = 1
for img in sorted(glob.glob(os.path.join(in_dir, "*.png"))):
    image = cv2.imread(img,cv2.IMREAD_COLOR)
    mask = ((image)*50) #brighten image to make mask
    mask = cv2.resize(mask, (img_w, img_h)) #resize images for uniformity
    os.chdir(out_dir)
    
    #all of this code is so that the images will be in proper ascending order
    if total_images < 100:
        if count < 10:
            cv2.imwrite("mask_0"+str(count)+".tif", mask) #add leading 0
            count += 1
        else:
            cv2.imwrite("mask_"+str(count)+".tif", mask)
            count += 1
            
    else: 
        if total_images >= 100 and total_images < 1000:
            if count < 10:
                cv2.imwrite("mask_00"+str(count)+".tif", mask) #add leading 00
                count += 1
            else: 
                if count >=10 and count <100:
                    cv2.imwrite("mask_0"+str(count)+".tif", mask)
                    count += 1
            

##save the masks off without alteration for machine learning
count2 = 1
for img in sorted(glob.glob(os.path.join(in_dir, "*.png"))):
    image = cv2.imread(img,cv2.IMREAD_COLOR)
    mask = cv2.resize(image, (img_w, img_h)) #resize images for uniformity
    os.chdir(mask_dir)
    
    #all of this code is so that the images will be in proper ascending order
    if total_images < 100:
        if count2 < 10:
            cv2.imwrite("mask_0"+str(count2)+".tif", mask) #add leading 0
            count2 += 1
        else:
            cv2.imwrite("mask_"+str(count2)+".tif", mask)
            count2 += 1
            
    else: 
        if total_images >= 100 and total_images < 1000:
            if count2 < 10:
                cv2.imwrite("mask_00"+str(count2)+".tif", mask) #add leading 00
                count2 += 1
            else: 
                if count2 >=10 and count2 <100:
                    cv2.imwrite("mask_0"+str(count2)+".tif", mask)
                    count2 += 1  

