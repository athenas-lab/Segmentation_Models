import os
import glob
import cv2
from PIL import Image
import pandas as pd
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2



#sys.path.append("../..")
#import data.data_utils as dut


""" 
Cambridge video frames driving dataset processing for multi-class segmentation (32 classes) using SegFormer.
https://www.kaggle.com/datasets/carlolepelaars/camvid
images: RGB
annotated masks: RGB
"""


class CamvidDataset(Dataset):
    """ Dataset processor for camvid training and test dataset """

    def __init__(self, image_proc, img_paths,  mask_paths=None, labeler=None, augment=False):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
       
        self.mask_labeler = labeler #maps classes to rgb labels
        self.image_proc = image_proc
        self.augment = augment

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]
        #load image
        img = cv2.imread(self.image_paths[idx]) #load as BGR image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #convert to RGB
        if self.mask_paths is not None:
            #load the mask view. 
            mask = cv2.imread(self.mask_paths[idx]) #load mask as BGR image
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  #convert to RGB
            #print(self.mask_paths[idx], mask.shape, mask[0, 0, :])
            #map the pixels in the bgr mask to grayscale labels
            #mask = self.mask_labeler.bgr2gray(mask) #same as below
            mask = self.mask_labeler.mapMaskToClassLabels(mask)
        else: mask = None    
        #print("before aug", mask.shape, mask[0,0])
        #perform necessary transformations
        if self.augment:
            if mask is not None: #for training data there are GT masks 
               img, mask = augmentImages(image=img, mask=mask) 
               
            else: #for test data there may not be any GT mask
               img = augmentImages(image=img)
               
        if mask is not None:
            #process the image and mask through the Segformer pre-trained image processer      
            encoded_inps = self.image_proc(img, mask, return_tensors="pt")
            #print("after aug", mask.shape, mask[0,0])
        else:    
            encoded_inps = self.image_proc(img, return_tensors="pt")
      
        for k, v in encoded_inps.items():
            encoded_inps[k].squeeze_()

        return encoded_inps


def augmentImages(image, mask):
    """ Return image transform function """
    
    # Albumentations for applying the same transforms on the img and mask.
    # Image and mask are  supplied as np array.
    
    aug = A.Compose(
      [
          A.Flip(p=0.5),
          A.RandomRotate90(p=0.5),
          A.OneOf([
                  A.Affine(p=0.33,shear=(-5,5),rotate=(-80,90)),
                  A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=120,
                    #border_mode= cv2.BORDER_CONSTANT,
                    #value=255, # padding with the ignored class 
                    p=0.33),
                  A.GridDistortion(p=0.33),
                ], p=1),
          A.CLAHE(p=0.8),
          A.OneOf(
              [
                  A.ColorJitter(p=0.33),
                  A.RandomBrightnessContrast(p=0.33),    
                  A.RandomGamma(p=0.33)
              ],
              p=1
          )
          ]
    )
    augmentation = aug(image=image, mask=mask)
    aug_img, aug_mask = augmentation['image'], augmentation['mask']
       
    return aug_img, aug_mask



def loadTrainingData(cfg, labeler, feature_extractor):
    """ Load the training and validation data """

    #training data 
    train_img_paths = os.listdir(cfg.train_path)
    train_mask_paths = []
    for idx, im in enumerate(train_img_paths):
        fn, ext = os.path.splitext(im)
        train_img_paths[idx] = os.path.join(cfg.train_path, fn + ".png")
        mask_path = os.path.join(cfg.train_mask_path, fn + "_L.png")
        train_mask_paths.append(mask_path)

    #validation data 
    val_img_paths = sorted(os.listdir(cfg.val_path))
    val_mask_paths = []
    for idx, im in enumerate(val_img_paths):
        fn, ext = os.path.splitext(im)
        val_img_paths[idx] = os.path.join(cfg.val_path, fn + ".png")
        mask_path = os.path.join(cfg.val_mask_path, fn + "_L.png")
        val_mask_paths.append(mask_path)


    #print(train_img_paths[:5], train_mask_paths[:5])
    #print(val_img_paths[:5], val_mask_paths[:5])
    print(len(train_img_paths), len(train_mask_paths))
    print(len(val_img_paths), len(val_mask_paths))
   
    #get the training dataset 
    train_ds = CamvidDataset(feature_extractor, train_img_paths, train_mask_paths, labeler, augment=True)
    #get the validation dataset
    val_ds = CamvidDataset(feature_extractor, val_img_paths, val_mask_paths, labeler)

    #Number of batches loaded in advance by each worker.
    #Default to 5, meaning 5*(num_workers) batches will be loaded in advance.
    prefetch = 5 
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, prefetch_factor=prefetch)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, prefetch_factor=prefetch)
    
    return train_dl, val_dl, train_ds, val_ds


def loadTestData(cfg, labeler, feature_extractor):
    """ Load the test data """

    #test data 
    test_img_paths = sorted(os.listdir(cfg.test_path))
    test_mask_paths = []
    for idx, im in enumerate(test_img_paths):
        fn, ext = os.path.splitext(im)
        test_img_paths[idx] = os.path.join(cfg.test_path, fn + ".png")
        mask_path = os.path.join(cfg.test_mask_path, fn + "_L.png")
        test_mask_paths.append(mask_path) 
  
    print(len(test_img_paths), len(test_mask_paths))
    #get the test dataset
    test_ds = CamvidDataset(feature_extractor, test_img_paths, test_mask_paths, labeler)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return test_dl
