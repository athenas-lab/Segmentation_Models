import os
import glob
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T

""" Salt segmentation dataset processing """



class SaltDataset(Dataset):

    def __init__(self, img_paths, mask_paths=None, transf=None):

        # store the image and mask filepaths, and transform function
        self.image_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transf

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
        if self.mask_paths is not None:
           #load the binary mask in grayscale mode
           mask = cv2.imread(self.mask_paths[idx], 0)

        #img = Image.open(self.image_paths[idx])
       
        #load the binary mask. Convert the mask to black and white
        #mask = Image.open(self.mask_paths[idx]).convert("L")

        #transform the image and mask
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
    """ Return image transform function """

    #apply same transform to training data (img, mask)
    train_transf = T.Compose(
           [ T.ToPILImage(),
             T.Resize((cfg.img_height, cfg.img_width)),
             T.RandomHorizontalFlip(p=0.5),
             T.RandomVerticalFlip(p=0.1),
             T.ToTensor(), #should come before Normalize. (HWC->CHW; /255) 
             #resultant values will be in [-1,1]
             T.Lambda(lambda t: (t*2) -1) # can be used on mask and image even when they have different channel dim 
           ]
          )

    #transform validation data 
    val_transf = T.Compose(
           [ T.ToPILImage(),
             T.Resize((cfg.img_height, cfg.img_width)),
             T.ToTensor(),
             #resultant values will be in [-1,1]
             T.Lambda(lambda t: (t*2) -1)
           ]
        )


    return train_transf, val_transf

def loadData(cfg):
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

    train_transf, val_transf = getTransform(cfg)

    #create the train and test datasets
    train_data = SaltDataset(train_imgs, train_masks, train_transf)
    val_data = SaltDataset(val_imgs, val_masks, val_transf)


    train_dl = DataLoader(train_data, shuffle=True, batch_size=cfg.batch_size, num_workers=os.cpu_count())
    val_dl = DataLoader(val_data, shuffle=False, batch_size=cfg.batch_size, num_workers=os.cpu_count())

    return train_dl, val_dl

def getTestData(cfg, gt_mask=False):
    """ Load the test data """

    img_paths = sorted(glob.glob(cfg.test_path + "/*.png"))

    #get the transforms
    _,  test_transf = getTransform(cfg)

    #get the test dataset
    test_ds = SaltDataset(img_paths, transf=test_transf)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    return test_dl


