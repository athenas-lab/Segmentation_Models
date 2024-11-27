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

""" DIVA virtual try-on image dataset processing """


class DivaDataset(Dataset):

    def __init__(self, img_paths, mask_paths=None, transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transf

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]
        #load image
        img = Image.open(self.image_paths[idx])

        if self.mask_paths is not None:
           #load the mask view. Convert the mask to black and white
           mask = Image.open(self.mask_paths[idx]).convert("L")
        
        #transform if needed
        if self.transform is not None:

            if self.mask_paths is not None: 
              #apply the same transformation to the image and mask 
              img, mask = self.transform(img, mask)
            
            else: #no GT mask for test data
              img = self.transform(img)

        if self.mask_paths is not None:
            sample = {"img_name": file_name, "image": img, "mask": mask}
        else: #no GT masks
           sample ={"img_name": file_name, "image": img}

        return  sample    


def getTransform(cfg):
    """ Return image transform functions """

    #apply same transform to training data (img, mask)
    train_transf = T.Compose(
           [ T.Resize((cfg.img_height, cfg.img_width)),
             T.RandomHorizontalFlip(p=0.5),
             T.RandomVerticalFlip(p=0.1),
             T.ToTensor(), #should come before Normalize. (HWC->CHW; /255) 
             #resultant values will be in [-1,1]
             T.Lambda(lambda t: (t*2) -1) # can be used on mask and image even when they have different channel dim 
           ]
          )  

    #transform validation data 
    val_transf = T.Compose(
           [ T.Resize((cfg.img_height, cfg.img_width)),
             T.ToTensor(),
             #resultant values will be in [-1,1]
             T.Lambda(lambda t: (t*2) -1)
           ]
        )
    

    return train_transf, val_transf



def loadData(cfg):
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
    #print(len(train_imgs), len(train_masks))
    #print(len(val_imgs), len(val_masks))
                
    #get the transforms
    train_transf, val_transf = getTransform(cfg)

    #get the training dataset 
    train_ds = DivaDataset(train_imgs, train_masks, train_transf)
    #get the validation dataset
    val_ds = DivaDataset(val_imgs, val_masks, val_transf)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    
    return train_dl, val_dl

def getTestData(cfg, gt_mask=False):
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
    _,  test_transf = getTransform(cfg)

    #get the test dataset
    test_ds = DivaDataset(img_paths, transf=test_transf)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
   
    return test_dl            
   

