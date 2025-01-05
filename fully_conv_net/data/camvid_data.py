import os
import glob
import cv2
from PIL import Image
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

""" 
Cambridge video frames dataset processing for multi-class image segmentation (32 classes).
https://www.kaggle.com/datasets/carlolepelaars/camvid
images: RGB
annotated masks: RGB
"""



class CamvidDataset(Dataset):

    def __init__(self, img_paths, img_transf, mask_paths=None, labeler=None, mask_transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transf
        self.mask_transform = mask_transf
        self.mask_labeler = labeler

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]
        #load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)
        #transform image if needed
        if self.img_transform is not None:

            #apply the same transformation to the image and mask 
            img = self.img_transform(image=img)["image"]
 
 
        if self.mask_paths is not None:
            #load the mask view. 
            mask = Image.open(self.mask_paths[idx]).convert("RGB")          
            mask = np.array(mask)
            #print("before transform", self.mask_paths[idx], mask.shape, mask[100, 100, :])

            
            #transform mask if needed
            if self.mask_transform is not None:
                mask = self.mask_transform(image=mask)["image"]
            #print("after transform", mask, mask.shape)
            #label the pixels of the seg map with class labels instead of rgb values   
            mask_labels = self.mask_labeler.mapMaskToClassLabels(mask)
            #print("after mapping to class labels", mask_labels.shape, mask, mask_labels[100,100])
            mask = torch.tensor(mask_labels, dtype=torch.long)  
     
        if self.mask_paths is not None:
           sample = {"img_name": file_name, "image": img, "mask": mask}
        else: #no GT masks
           sample ={"img_name": file_name, "image": img} 

        return  sample    


def getTransform(cfg):
    """ Return image transform function """

    
    # Albumentations for applying the same transforms on the img and mask.
    # Image is supplied as np array. However, the mask looks black.
    img_transf = A.Compose(
           [ A.Resize(cfg.img_height, cfg.img_width),
            #A.Rotate(limit = 35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            #resultant values will be in [0, 1]
            #A.Normalize(
            #    mean=[0.0, 0.0 , 0.0],
            #    std=[1.0, 1.0, 1.0]
            #),
            A.Normalize(
                mean=[0.45734706, 0.43338275, 0.40058118],
                std=[0.23965294, 0.23532275, 0.2398498]
            ),
            ToTensorV2()
           ])
       
    mask_transf = A.Compose(
           [ A.Resize(height=cfg.img_height, width=cfg.img_width)
           ]
          )   

    
    """
    #apply same transform to training data (img, mask)
    train_transf = T.Compose(
           [ T.Resize((cfg.img_height, cfg.img_width)),
             T.RandomHorizontalFlip(p=0.5),
             T.RandomVerticalFlip(p=0.1),
             T.ToTensor(), #should come before Normalize. (HWC->CHW; /255) 
             #resultant values will be in [-1,1]
             #T.Lambda(lambda t: (t*2) -1) # can be used on mask and image even when they have different channel dim 
             #T.Normalize( #cannot be used on grayscale masks
             #   mean=(0.5, 0.5 , 0.5),
             #   std=(0.5, 0.5, 0.5)
            #)
           ]
          )  

    #transform validation data 
    val_transf = T.Compose(
           [ T.Resize((cfg.img_height, cfg.img_width)),
             T.ToTensor(),
             #resultant values will be in [-1,1]
             #T.Lambda(lambda t: (t*2) -1)
             #T.Normalize(
             #   mean=(0.5, 0.5 , 0.5),
             #   std=(0.5, 0.5, 0.5)
            #)
           ]
        )
    """

    return img_transf, mask_transf



def loadTrainingData(cfg, labeler):
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

   
    #print(train_img_paths[:5], train_mask_paths[:5])
    #print(val_img_paths[:5], val_mask_paths[:5])
    print(len(train_img_paths), len(train_mask_paths))
    print(len(val_img_paths), len(val_mask_paths))
                
    #get the transforms
    img_transf, mask_transf = getTransform(cfg)
    
    #get the training dataset 
    train_ds = CamvidDataset(train_img_paths, img_transf, train_mask_paths, labeler, mask_transf)
    #get the validation dataset
    val_ds = CamvidDataset(val_img_paths, img_transf, val_mask_paths, labeler, mask_transf)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    
    return train_dl, val_dl

def loadTestData(cfg, labeler, gt_mask=False):
    """ Load the test data """
 
    img_paths = sorted(os.listdir(cfg.test_path))
    mask_paths = [] if gt_mask else None
    for idx, im in enumerate(img_paths):
        fn, ext = os.path.splitext(im)
        img_paths[idx] = os.path.join(cfg.test_path, fn + ".png")
        if gt_mask:
           mask_path = os.path.join(cfg.test_mask_path, fn + "_L.png")
           mask_paths.append(mask_path)

    #get the transforms
    img_transf,  mask_transf = getTransform(cfg)

    #get the test dataset
    test_ds = CamvidDataset(img_paths, img_transf, mask_paths, labeler, mask_transf)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
   
    return test_dl            
   
