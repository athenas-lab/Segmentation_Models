import os
import glob
import cv2
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
DIVA virtual try-on dataset processing for binary image segmentation (clothes/not clothes).
images: RGB
annotated masks: black and white with pixel value = 0 or 255.
"""

class DivaDatasetBCE(Dataset):
   """ Dataset processing for Binary Cross-Entropy """

    def __init__(self, img_paths, img_transf, mask_paths=None,  mask_transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transf
        self.mask_transform = mask_transf

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]
        #load image
        img = Image.open(self.image_paths[idx])
        img = np.array(img)
        #transform image if needed
        if self.img_transform is not None:

            #apply transformation to the image 
            img = self.img_transform(image=img)["image"]
  
        if self.mask_paths is not None:
            #load the mask view. 
            mask = Image.open(self.mask_paths[idx]).convert("L")          
            mask = np.array(mask)
            #print("before transform", self.mask_paths[idx], mask.shape, img.shape, img[:, 100, 100], mask[100, 100], mask.min(), mask.max())
                       
            #transform mask if needed
            if self.mask_transform is not None:
                mask = self.mask_transform(image=mask)["image"]
                mask = mask[np.newaxis, :, :] #BCE loss requires a channel dim
            #print("after transform", mask, mask.shape, mask.min(), mask.max())
            #pixel values are 0 or 1 with floating point type for compatibility with BCELogitsLoss  
            mask_labels = (mask == mask.max()) *1.0     
            mask = torch.tensor(mask_labels)  
            #print("after mapping to class labels", mask, mask.shape, mask_labels.shape, mask_labels[:, 100,100], mask.min(), mask.max())
            sample = {"img_name": file_name, "image": img, "mask": mask}
        else: #no GT masks
          sample ={"img_name": file_name, "image": img} 

        return  sample    

class DivaDatasetCE(Dataset):
    """ Dataset processing for Categorical Cross-Entropy """
    def __init__(self, img_paths, img_transf, mask_paths=None,  mask_transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transf
        self.mask_transform = mask_transf

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]
        #load image
        img = Image.open(self.image_paths[idx])#.convert("RGB")
        img = np.array(img)
        #print(img, img.shape, img.min(), img.max())
        #transform image if needed
        if self.img_transform is not None:

            #apply transformation to the image 
            img = self.img_transform(image=img)["image"]
            
 
        if self.mask_paths is not None:
            #load the mask view. 
            mask = Image.open(self.mask_paths[idx]).convert("L")          
            mask = np.array(mask)
            #print("before transform", self.mask_paths[idx], mask.shape, img.shape, img[:, 100, 100], img.min(), img.max(), mask[100, 100], mask.min(), mask.max())        
            #transform mask if needed
            if self.mask_transform is not None:
                mask = self.mask_transform(image=mask)["image"]
               
            #print("after transform", mask, mask.shape, mask.min(), mask.max())
            #label the pixels of the seg map with class labels instead of rgb values   
            mask_labels = (mask == mask.max()) *1     
            mask = torch.tensor(mask_labels, dtype=torch.long)  
            #print("after mapping to class labels", mask, mask.shape, mask_labels.shape, mask_labels[100,100], mask.min(), mask.max())
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
            ToTensorV2(),
           ])
       
    mask_transf = A.Compose(
           [ A.Resize(height=cfg.img_height, width=cfg.img_width)
           ]
          )   

    return img_transf, mask_transf



def loadTrainingData(cfg, labeler):
    """ Load the training and validation data """
 
    img_paths = os.listdir(cfg.train_path)
    mask_paths = []
    for idx, im in enumerate(img_paths):
        fn, ext = os.path.splitext(im)
        img_paths[idx] = os.path.join(cfg.train_path, fn + ".png")
        mask_path = os.path.join(cfg.mask_path, fn + ".png")
        mask_paths.append(mask_path)

    #print(img_paths[:5], mask_paths[:5])

    splits = train_test_split(img_paths, mask_paths,
                        test_size=cfg.test_split, random_state=42)
    (train_imgs, val_imgs) = splits[:2]
    (train_masks, val_masks) = splits[2:]
    
    #print(train_imgs[:5], train_masks[:5])
    #print(val_imgs[:5], val_masks[:5])
    print("Train data:", len(train_imgs), len(train_masks))
    print("Val data:", len(val_imgs), len(val_masks))
                
                  
    #get the transform function for images and masks
    img_transf, mask_transf = getTransform(cfg)
    if cfg.loss_fn == "bce":            
        #get the training dataset 
        train_ds = DivaDatasetBCE(train_imgs, img_transf, train_masks, mask_transf)
        #get the validation dataset
        val_ds = DivaDatasetBCE(val_imgs, img_transf, val_masks, mask_transf)
 
    elif cfg.loss_fn == "ce":     
        #get the training dataset 
        train_ds = DivaDatasetCE(train_imgs, img_transf, train_masks, mask_transf)
        #get the validation dataset
        val_ds = DivaDatasetCE(val_imgs, img_transf, val_masks, mask_transf)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    
    return train_dl, val_dl

   
def loadTestData(cfg, labeler=None, gt_mask=False):
    """ Load the test data """
 
    img_paths = sorted(os.listdir(cfg.test_path))
    mask_paths = [] if gt_mask else None
    for idx, im in enumerate(img_paths):
        fn, ext = os.path.splitext(im)
        img_paths[idx] = os.path.join(cfg.test_path, fn + ".png")
        if gt_mask:
           mask_path = os.path.join(cfg.mask_path, fn + ".png")
           mask_paths.append(mask_path)
      
    #get the transforms
    img_transf, mask_transf = getTransform(cfg)
    if cfg.loss_fn == "bce":            
        #get the test dataset
        test_ds = DivaDatasetBCE(img_paths, img_transf, mask_paths, mask_transf)
    elif cfg.loss_fn == "ce":                    
        #get the test dataset
        test_ds = DivaDatasetCE(img_paths, img_transf, mask_paths, mask_transf)
   
   
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
   
    return test_dl            
   

