import cv2
import numpy as np
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt

""" Data utility functions """

class SegMapClassLabeler:
    """ Used to map classes to RGB and vice-versa """

    def __init__(self, label_info):


        if type(label_info) == str:  #multi-class labels defined in a file 
           self.loadLabelsFromFile(label_info)
           #class ids to class name mapping
           self.id2label = {i:c for i, c in enumerate(self.classes)}
        elif type(label_info) == dict: #binary class mapping
            self.classes = list(label_info.values()) 
            self.class_rgb = []
            self.class_rgb.append((0, 0, 0)) 
            self.class_rgb.append((255, 255, 255)) 
            #class ids to class name mapping
            self.id2label = label_info

        self.num_classes = len(self.classes)
        #class name to class id mapping
        self.label2id = {v: k for k, v in self.id2label.items()}

    def loadLabelsFromFile(self, label_path):
        """ Get the segmentation classes and their labels """

        self.class_rgb = [] #mapping from color in segmentation map to class
        self.classes = []  #classes represented in the map

        with open(label_path, "r") as fp:
            for i, ln in enumerate(fp):
                if i == 0: continue #skip header
                class_name, r, g, b = ln.strip().split(",") 
                self.class_rgb.append((int(r), int(g),int(b)))
                self.classes.append(class_name)    
            #print(self.class_rgb, self.classes)    

        return self.classes, self.class_rgb


    def mapMaskToClassLabels(self, mask, bgr=False):
        """ 
        Map the pixels in the segmentation map to class labels 
        for use with categorical cross-entropy loss (Unet).
        
        """

        mask_labels = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for i, rgb in enumerate(self.class_rgb):
            label = np.array(rgb)
            if bgr: #reverse the input mask if BGR (if mask has been opened using cv2)
                label = label[::-1]
            #print(label)
            mask_labels[np.where(np.all(mask == label, axis=-1))[:2]] = i
        mask_labels = mask_labels.astype(int)    

        return mask_labels

    def bgr2gray(self, bgr_mask):
        """ 
        Convert BGR mask labels to grayscale labels for each pixel. 
        This is needed when the image and pixels are being read as BGR images by cv2.
        (Same as above with bgr=True).
        """

        #(H,W, C) --> (H, W, num_classes): vectorize to create binary pixel label for each class
        mask_shape = bgr_mask.shape[:-1] + (self.num_classes, )
        mask_label = np.zeros(mask_shape)

        #rgb label --> bgr --> vectorized to grayscale mask label for each class    
        #Each pixel has a label vector whose length = number of classes,
        #where label[i] = 1 if that pixel has a class label=i, else 0.
        for i, rgb in enumerate(self.class_rgb):
            bgr_col = rgb[::-1] #reverse rgb label to get bgr
            mask_label[:,:,i] = np.all(bgr_mask == bgr_col, axis=-1) * 1
        
        #assign the class ID. 
        mask_label = np.argmax(mask_label, axis=-1)    
        
        return mask_label


    def classToRGBMaskComplex(self, seg_labels):
        """ 
        Given a multiclass semantic segmentation map with pixels labeled with class ids, 
        convert the pixel class labels to RGB values corresponding to the classes and output the colored seg map.
        """
        
        red = np.zeros_like(seg_labels).astype(np.uint8)
        green = np.zeros_like(seg_labels).astype(np.uint8)
        blue = np.zeros_like(seg_labels).astype(np.uint8)

        #replace class labels with RGB values for each pixel
        for class_id in range(len(self.class_rgb)):
            idx = seg_labels == class_id #locate where the predicted mask values match the current class label 
            if idx is not None:
                red[idx] = np.array(self.class_rgb)[class_id, 0]
                green[idx] = np.array(self.class_rgb)[class_id, 1]
                blue[idx] = np.array(self.class_rgb)[class_id, 2]
                
        #output the colored seg map        
        seg_map = np.stack([red, green, blue], axis=2)
        #following does not produce good result when saving as standalone, but is needed for overlay
        #seg_map = np.array(seg_map, dtype=np.float32)  
        #print(f"labelsToSegMap: seg map shape: {seg_map.shape}")
        
        return seg_map

    def classToRGBMask(self, mask):
        """ Maps mask with class labels for 32 classes to RGB mask"""

        #adds the RGB channel: (H,W) + (3,) ==> (H, W, 3)
        rgb_mask_shape = mask.shape + (3, )
        #create a RGB mask with same H,W as input mask
        rgb_mask = np.zeros(rgb_mask_shape, dtype=np.uint8)

        for label, color in enumerate(self.class_rgb):
            rgb_mask[mask == label] = color

        return rgb_mask


def tensor2Img(x):
    """ Given a torch tensor, convert it to an image"""

    #define the image transforms
    img_transf =  T.Compose([     
             T.Lambda(lambda t: t.permute(1,2,0)), #CHW --> HWC
             #T.Lambda(lambda t: (t +1)/2.0), 
             T.Lambda(lambda t: t*255.), #RGB
             T.Lambda(lambda t: t.numpy().astype(np.uint8)), #convert to img
             T.ToPILImage()])

    z = img_transf(x)         

    return z

def tensor2ImgMeanStd(x):
    """ Given a torch tensor, convert it to an unnormalized image """

    mean= np.array([0.45734706, 0.43338275, 0.40058118])
    std= np.array([0.23965294, 0.23532275, 0.2398498])
    
    #define the image transforms
    img_transf =  T.Compose([     
             T.Lambda(lambda t: t.permute(1,2,0)), #CHW --> HWC
             T.Lambda(lambda t: t.numpy() * std + mean)]) #unnormalize
            ## T.ToPILImage()])

    z = img_transf(x)         

    return z


def getSegformerFeatureExtractor(cfg, model_ckpt, resize):
    """ Feature extractor for Segformer """

    from transformers import SegformerImageProcessor

    # image and mask are prepared using pretrained processer for training  with Segformer
    feature_extractor =  SegformerImageProcessor.from_pretrained(model_ckpt)
    feature_extractor.do_reduce_labels = cfg.reduce_labels
    feature_extractor.do_resize = resize
    if cfg.data_name == "camvid":
      feature_extractor.size = {"height":cfg.orig_img_h, "width":cfg.orig_img_w} #for camvid
    else:  
      feature_extractor.size = {"height":cfg.img_height, "width":cfg.img_width}
    feature_extractor.do_normalize= False
    feature_extractor.do_rescale= True
    
    return feature_extractor



