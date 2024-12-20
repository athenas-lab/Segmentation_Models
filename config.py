import os
from pathlib import Path
import torch

""" Configure hyper-parameters and settings """

class Config:

     def __init__(self, data_name, model_name, mode="train"):

        self.num_workers = 1 #number of worker threads for data loading
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
       
        self.data_name = data_name  
        self.debug = False
        
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
        self.model_name = model_name  #["segformer_ft_from_cityscapes1024", "unet_seg_bn", "unet_original"]
        print(f"Model Name: {self.model_name}")
        #threshold for pixel classification in mask
        self.threshold = 0.5
        #track progress every N steps
        self.save_and_sample_freq = 300 #500
        #folder to log results
        self.results_path = "./results/%s/%s/"%(data_name, self.model_name)
        if mode == "train":
           os.makedirs(self.results_path, exist_ok = True)
        self.model_path = "./saved_models/%s/%s"%(data_name, self.model_name)
        if mode == "train":
           os.makedirs(self.model_path, exist_ok = True)
        print(self.results_path, self.model_path) 

     def configSaltData(self):
           #salt segmentation dataset (black and white)
           
           data_root = "./datasets/segmentation/%s"%self.data_name
           self.orig_img_h  = 101 #original image size before resizing for RLE computation
           self.orig_img_w  = 101
           self.img_height = 128 
           self.img_width = 128 
           self.nchannels = 1
           self.num_classes = 1 #salt/no salt: binary classification
           self.test_split = 0.15
           self.train_path = os.path.join(data_root, "train/images")
           self.test_path = os.path.join(data_root,  "test/images")
           self.mask_path = os.path.join(data_root,  "train/masks")
           self.label_info = {0:"background", 1: "salt"}

           #------training config------
           #batch size for training      
           self.batch_size = 128
           #learning rate 
           self.init_lr = 0.001 
           #number of training epochs
           self.nepochs = 50 #100 for unet 

           #------model config------
           #UNet
           #encoder channels 
           self.enc_ch = [3, 16, 32, 64]
           #decoder channels 
           self.dec_ch = [64, 32, 16]
           #number of levels in Unet model
           self.num_levels = 4 
           #norm
           self.norm = "bn" #bn for batchnorm, else ""

           #segformer
           self.reduce_labels = False
           self.ignore_index = 255 #class id to ignore 
           return
     

     def configCamVidData(self):
           #camvid segmentation dataset for autonomous driving (RGB data, multiclass)
 
           data_root = "./datasets/segmentation/%s"%self.data_name
           self.img_height = 224  #resized height and width
           self.img_width = 224
           self.orig_img_h = 360
           self.orig_img_w = 480
           self.nchannels = 3
           self.num_classes = 32

           #images
           self.train_path = os.path.join(data_root, "train")
           self.test_path = os.path.join(data_root, "test")
           self.val_path = os.path.join(data_root, "val")
           #mask paths
           self.train_mask_path = os.path.join(data_root, "train_labels")
           self.test_mask_path = os.path.join(data_root, "test_labels")
           self.val_mask_path = os.path.join(data_root, "val_labels")
           self.label_info = os.path.join(data_root, "class_dict.csv")

           #------training config------
           #batch size for training
           self.batch_size = 16
           #learning rate 
           self.init_lr = 0.0001 
           #number of training epochs
           self.nepochs = 100

           #------model config------
           #Unet  model config
           #encoder channels for unet
           self.enc_ch = [3, 16, 32, 64]
           #decoder channels for unet
           self.dec_ch = [64, 32, 16]
           #number of levels in Unet model
           self.num_levels = 4 
           #norm
           self.norm = "bn" #bn for batchnorm, else ""

           #segformer model config
           #set this to True if  b/g class = 0 is included in the annotated masks  but not in list of classes,
           #in which case the class ids are reduced by 1 in the annotated map and the b/g label=0 is replaced by 255.
           #In the case of binary class, this reduces the number of classes to 1, in which BCELogitsLOss will need to be 
           #used. Otherwise, CrossEntropyLoss will be used. 
           self.reduce_labels=False 
           self.ignore_index=255 #class id to ignore 

           return

     def configCarvanaData(self):
           #carvana segmentation dataset for cars (RGB data, binary)

           data_root = "./datasets/segmentation/%s"%self.data_name
           self.orig_img_h  = 1280 #original image size before resizing for RLE computation
           self.orig_img_w  = 1918
           self.img_height = 320 #size after resizing
           self.img_width = 480
           self.nchannels = 3
           self.num_classes = 1 #for BCELogitsLoss, 2 for Segformer, cross-entropy loss
           self.test_split = 0.20
           self.train_path = os.path.join(data_root, "train")
           self.test_path = os.path.join(data_root,  "test")
           self.mask_path = os.path.join(data_root,  "train_masks")
           self.label_info = {0: "background", 1: "car"}

           #------training config------
           #batch size for training
           self.batch_size = 16
           #learning rate 
           self.init_lr = 0.001 
           #number of training epochs
           self.nepochs = 50

           #------model config------
           #Unet
           #encoder channels 
           self.enc_ch = [3, 64, 128, 256, 512]
           #decoder channels 
           self.dec_ch = [512, 256, 128, 64]
           #number of levels in Unet model
           self.num_levels = 4 
           #norm layers
           self.norm = "bn" #bn for batchnorm

           #segformer
           self.reduce_labels = False
           self.ignore_index = 255 #class id to ignore 

           return

     def configDivaData(self):
           #Diva segmentation dataset for clothes (RGB data, binary)

           data_root = "./datasets/vton/diva/dataset/"
           self.img_height = 720
           self.img_width = 540
           self.nchannels = 3
           self.num_classes = 1 #for BCELogitsLoss, 2 for Segformer, cross-entropy loss
           self.test_split = 0.20
           self.train_path = os.path.join(data_root, "cloth")
           self.mask_path = os.path.join(data_root,  "cloth_mask")
           self.label_info = {0: "background", 1: "cloth"}

           #------training config------
           #batch size for training
           self.batch_size = 4
           #learning rate 
           self.init_lr = 0.001 
           #number of training epochs
           self.nepochs = 1
           
           #------model config------
           #Unet config
           #encoder channels 
           self.enc_ch = [3, 64, 128, 256, 512]
           #decoder channels 
           self.dec_ch = [512, 256, 128, 64]
           #number of levels in Unet model
           self.num_levels = 4          
           #norm
           self.norm = "bn" #bn for batchnorm

           #segformer
           self.reduce_labels = False
           self.ignore_index = 255 #class id to ignore 

           return
