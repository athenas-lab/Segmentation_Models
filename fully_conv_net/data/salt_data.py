import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

""" 
Salt segmentation dataset processing for binary image segmentation.
Based on https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
"""

class SaltDatasetBCE(Dataset):
   """ Dataset processing for Binary Cross-Entropy """

    def __init__(self, img_paths, img_transf,  mask_paths=None, mask_transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transf
        self.mask_transform = mask_transf

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #load the sample img and mask and return 

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]

        #load the image and mask
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         
        #img = Image.open(self.image_paths[idx])
       
        #load the binary mask. Convert the mask to black and white
        #mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.img_transform is not None:
            #apply transformation to the image 
            img = self.img_transform(image=img)["image"]
 
        if self.mask_paths is not None:
            #load the binary mask in grayscale mode
            mask = cv2.imread(self.mask_paths[idx], 0)
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

        if self.mask_paths is not None:
           sample = {"img_name": file_name, "image": img, "mask": mask}
        else: #no GT masks
           sample ={"img_name": file_name, "image": img} 

        return  sample

class SaltDatasetCE(Dataset):
    """ Dataset processing for Categorical Cross-Entropy """
    
    def __init__(self, img_paths, img_transf,  mask_paths=None, mask_transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transf
        self.mask_transform = mask_transf

    def __len__(self):
        #return the number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):

        #load the sample img and mask and return 

        #get the file name. used mainly for kaggle submission for testing
        file_name = self.image_paths[idx].split("/")[-1]

        #load the image and mask
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
        #img = Image.open(self.image_paths[idx])
       
        #load the binary mask. Convert the mask to black and white
        #mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.img_transform is not None:
            #apply transformation to the image 
            img = self.img_transform(image=img)["image"]
 
        if self.mask_paths is not None:
            #load the binary mask in grayscale mode
            mask = cv2.imread(self.mask_paths[idx], 0)
            #print("before transform", self.mask_paths[idx], mask.shape, mask[100, 100], mask.min(), mask.max())
                       
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
            ToTensorV2()
           ])
       
    mask_transf = A.Compose(
           [ A.Resize(height=cfg.img_height, width=cfg.img_width)
           ]
          )   

    return img_transf, mask_transf

def loadTrainingData(cfg, labeler=None):
    """ Get the data loaders for training on the Salt segmentation data """

    img_paths = glob.glob(cfg.train_path + "/*.png")
    mask_paths = [x.replace("images", "masks") for x in img_paths]
    print(len(img_paths), len(mask_paths))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(img_paths, mask_paths,
    	test_size=cfg.test_split, random_state=42)
    # unpack the data split
    (train_imgs, val_imgs) = split[:2]
    (train_masks, val_masks) = split[2:]
    print(len(train_imgs), len(train_masks))

    img_transf, mask_transf = getTransform(cfg)
    if cfg.loss_fn == "bce":
    
        #create the train and test datasets
        train_data = SaltDatasetBCE(train_imgs, img_transf, train_masks, mask_transf)
        val_data = SaltDatasetBCE(val_imgs, img_transf, val_masks, mask_transf)

    
    elif cfg.loss_fn == "ce":
       
        #create the train and test datasets
        train_data = SaltDatasetCE(train_imgs, img_transf, train_masks, mask_transf)
        val_data = SaltDatasetCE(val_imgs, img_transf, val_masks, mask_transf)


    train_dl = DataLoader(train_data, shuffle=True, batch_size=cfg.batch_size, num_workers=os.cpu_count())
    val_dl = DataLoader(val_data, shuffle=False, batch_size=cfg.batch_size, num_workers=os.cpu_count())

    return train_dl, val_dl

def loadTestData(cfg, gt_mask=False):
    """ Load the test data """

    img_paths = sorted(glob.glob(cfg.test_path + "/*.png"))

    #get the transforms
    img_transf, mask_transf = getTransform(cfg)
    if cfg.loss_fn == "bce":
        #get the test dataset
        test_ds = SaltDatasetBCE(img_paths, img_transf, mask_paths, mask_transf)
    elif cfg.loss_fn == "ce":
        #get the test dataset
        test_ds = SaltDatasetCE(img_paths, img_transf, mask_paths, mask_transf)

    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    return test_dl


