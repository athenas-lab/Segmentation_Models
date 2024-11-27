import os
from pathlib import Path
import torch

""" Configure hyper-parameters and settings """

class Config:

     def __init__(self, data_name, mode="train"):

        self.num_workers = 4 #number of worker threads for data loading
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
       
        self.data_name = data_name     
        
        """ Dataset-specific configuration """
        if data_name == "salt":
           self.configSaltData()
        elif data_name == "camvid":
           self.configCamVidData()
        elif data_name == "carvana":
           self.configCarvanaData()
        elif data_name == "diva":
           self.configDivaData()

        #training config
        model_name = "unet_seg_bn" 
        print(f"Model Name: {model_name}")
        #threshold for pixel classification in mask
        self.threshold = 0.5
        #track progress every N steps
        self.save_and_sample_freq = 300 #500
        #folder to log results
        self.results_path = "./results/%s/%s/"%(model_name, data_name)
        if mode == "train":
           os.makedirs(self.results_path, exist_ok = True)
        self.model_path = "./saved_models/%s/%s"%(model_name, data_name)
        if mode == "train":
           os.makedirs(self.model_path, exist_ok = True)


     def configSaltData(self):
           #salt segmentation dataset (black and white)
           
           data_root = "./datasets/segmentation/"
           self.orig_img_h  = 101 #original image size before resizing for RLE computation
           self.orig_img_w  = 101
           self.img_height = 128 
           self.img_width = 128 
           self.nchannels = 1
           self.num_classes = 1 #salt/no salt: binary classification
           self.test_split = 0.15
           self.train_path = os.path.join(data_root,  self.data_name, "train/images")
           self.test_path = os.path.join(data_root,  self.data_name, "test/images")
           self.mask_path = os.path.join(data_root,  self.data_name, "train/masks")

           #batch size for training
           self.batch_size = 128
           #learning rate 
           self.init_lr = 0.001 
           #encoder channels 
           self.enc_ch = [3, 16, 32, 64]
           #decoder channels 
           self.dec_ch = [64, 32, 16]
     
           #number of levels in Unet model
           self.num_levels = 4 
           #number of training epochs
           self.nepochs = 100
           #norm
           self.norm = "bn" #bn for batchnorm, else ""

           return
     


     def configCarvanaData(self):
          #carvana segmentation dataset for cars (RGB data, binary)

           data_root = "./datasets/segmentation/"
           self.orig_img_h  = 1280 #original image size before resizing for RLE computation
           self.orig_img_w  = 1918
           self.img_height = 320 #size after resizing
           self.img_width = 480
           self.nchannels = 3
           self.num_classes = 1
           self.test_split = 0.20
           self.train_path = os.path.join(data_root,  self.data_name, "train")
           self.test_path = os.path.join(data_root,  self.data_name, "test")
           self.mask_path = os.path.join(data_root,  self.data_name, "train_masks")

           #batch size for training
           self.batch_size = 16
           #learning rate 
           self.init_lr = 0.001 
           #encoder channels 
           self.enc_ch = [3, 64, 128, 256, 512]
           #decoder channels 
           self.dec_ch = [512, 256, 128, 64]

           #number of levels in Unet model
           self.num_levels = 4 
           #number of training epochs
           self.nepochs = 100
           #norm
           self.norm = "bn" #bn for batchnorm

           return

     def configDivaData(self):
           #Diva segmentation dataset for clothes (RGB data, binary)

           data_root = "./datasets/segmentation/"
           self.img_height = 720
           self.img_width = 540
           self.nchannels = 3
           self.num_classes = 1
           self.test_split = 0.20
           self.train_path = os.path.join(data_root, "cloth")
           self.mask_path = os.path.join(data_root,  "cloth_mask")

           #batch size for training
           self.batch_size = 8
           #learning rate 
           self.init_lr = 0.001 
           #encoder channels 
           self.enc_ch = [3, 64, 128, 256, 512]
           #decoder channels 
           self.dec_ch = [512, 256, 128, 64]

           #number of levels in Unet model
           self.num_levels = 4 
           #number of training epochs
           self.nepochs = 100
           #norm
           self.norm = "bn" #bn for batchnorm

           return
