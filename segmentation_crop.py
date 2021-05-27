"""
@author: Jeremy Harris
@date: 12/3/2020
"""
#########################################################################
#This code crops the images to reduce the image size by cropping out
#the background. Hopefully, we will see an improved training and test
#speed for results.
#########################################################################

from PIL import Image
import glob
import cv2
import os

#setup folder locations for input and output
os.chdir("/root/mlhome/segmentation/") #change directory
images_dir = './temp_frames/'
masks_dir = './temp_mask/'

images_out_dir = './cropped_images/'
masks_out_dir = './cropped_masks/'

#crop image dimensions
left = 0
top = 165
right = 2262
bottom = 530

#read in images & crop
files_in_folder = os.listdir(images_dir)
for img in files_in_folder:
    image = Image.open(images_dir+img)
    cropped = image.crop((left, top, right, bottom))
    cropped.save(images_out_dir+img)

#read in masks & crop
masks_in_folder = os.listdir(masks_dir)
for msk in masks_in_folder:
    mask = Image.open(masks_dir+msk)
    cropped_m = mask.crop((left, top, right, bottom))
    cropped_m.save(masks_out_dir+msk)

