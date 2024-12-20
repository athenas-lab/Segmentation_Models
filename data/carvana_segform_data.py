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


""" 
Carvana image dataset processing for binary image segmentation (car or not).
https://www.kaggle.com/competitions/carvana-image-masking-challenge
images: RGB
annotated masks: black and white with pixel value = 0 or 1
"""

class CarvanaDataset(Dataset):
    """ Dataset processor for carvana training and test dataset """

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
        img = Image.open(self.image_paths[idx]) 
        #img =  cv2.imread(self.image_paths[idx])
        if self.mask_paths is not None:
            #load the mask view. Pixel values are already either 0 and 1 and so dont need to be mapped to class IDs. 
            mask = Image.open(self.mask_paths[idx])
            #print(self.mask_paths[idx])
            #print(self.mask_paths[idx], mask.shape, mask[250, 250, :])
            #map the pixels in the bgr mask to grayscale labels
            #mask = self.mask_labeler.bgr2gray(mask) #same as below
            #mask = self.mask_labeler.mapMaskToClassLabels(mask, bgr=True)
        else: mask = None   
        
        #pixels = list(mask.getdata())
        #print(pixels)
        #print(img.size, mask.size, min(pixels), max(pixels))
        #mask.convert("L").show()
        

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

    img_paths = os.listdir(cfg.train_path)
    mask_paths = []
    for idx, im in enumerate(img_paths):
        fn, ext = os.path.splitext(im)
        img_paths[idx] = os.path.join(cfg.train_path, fn + ".jpg")
        mask_path = os.path.join(cfg.mask_path, fn + "_mask.gif")
        mask_paths.append(mask_path)

    #print(img_paths[:5], mask_paths[:5])

    splits = train_test_split(img_paths, mask_paths,
                        test_size=cfg.test_split, random_state=42)
    (train_imgs, val_imgs) = splits[:2]
    (train_masks, val_masks) = splits[2:]

    #print(train_imgs[:5], train_masks[:5])
    #print(val_imgs[:5], val_masks[:5])
    print(len(train_imgs), len(train_masks))
    print(len(val_imgs), len(val_masks))

    #get the training dataset 
    train_ds = CarvanaDataset(feature_extractor, train_imgs, train_masks, labeler)
    #get the validation dataset
    val_ds = CarvanaDataset(feature_extractor, val_imgs, val_masks, labeler)

   
    #Number of batches loaded in advance by each worker.
    #Default to 5, meaning 5*(num_workers) batches will be loaded in advance.
    prefetch = 1
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, prefetch_factor=prefetch)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, prefetch_factor=prefetch)
    
    return train_dl, val_dl, train_ds, val_ds


def loadTestData(cfg, labeler, feature_extractor, gt_mask=False):
    """ Load the test data """

    #test data 
    test_img_paths = sorted(os.listdir(cfg.test_path))[:100]
    test_mask_paths = [] if gt_mask else None

    for idx, im in enumerate(test_img_paths):
        fn, ext = os.path.splitext(im)
        test_img_paths[idx] = os.path.join(cfg.test_path, fn + ".jpg")
        if gt_mask: #if GT mask labels exist for the test data 
            mask_path = os.path.join(cfg.test_mask_path, fn + "_mask.gif")
            test_mask_paths.append(mask_path) 
    
    print("number of test images", len(test_img_paths))
    #get the test dataset
    test_ds = CarvanaDataset(feature_extractor, test_img_paths, test_mask_paths, labeler)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    
    return test_dl
