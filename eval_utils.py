import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torcheval.metrics.functional import binary_f1_score, multiclass_f1_score

import data.salt_data as dataset 

def displayPerf(hist):

    # plot the loss values over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(hist["train"], label="train_loss")
    plt.plot(hist["val"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    #plt.savefig(config.PLOT_PATH)
    plt.show()

    return

def preparePlot(cfg, imgs, preds, gt_masks):
    """ 
    Plot the segmentation masks with ground-truth.

    imgs, preds, gt_masks are numpy image arrays. 
    
    """
   
    pred_masks = []
    for i in range(len(preds)):
        #filter out the weak predictions. 
        #pixels with predictions <= threshold are mapped to 0 (black), correspond to absence of class. (for binary class)
        #pixels with predictions > threshold are mapped to 1 (white), correspond to presence of class. (for binary class)
        pred_mask = (preds[i] > cfg.threshold) * 255
        pred_mask = pred_mask.astype(np.uint8)
        pred_masks.append(pred_mask)

    #number of rows
    nr = 3 if gt_masks is not None else 2
    #initialize our figure
    figure, axs = plt.subplots(nrows=3, ncols=len(imgs), figsize=(30, 30))

    axs[0,0].set_title("Image")
    axs[1,0].set_title("Predicted Mask")
    if gt_masks is not None:
       axs[2,0].set_title("Ground-Truth Mask")
    for i in range(len(imgs)):
        # plot the original image, its mask, and the predicted mask
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(pred_masks[i])
        if gt_masks is not None:
           axs[2, i].imshow(gt_masks[i])
        # set the titles of the subplots
        #ax[2].set_title("Ground Truth Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.show()
    #figure.show()
    return



def computeSegMetricsTorch(gt, preds, num_class):
    """
        Compute segmentation metrics.
        gt: ground-truth masks as torch tensor
        preds: predicted masks as torch tensor
        num_class: number of semantic classes

    """

    assert (gt.shape == preds.shape)
    #intersection and union
    inter = (preds* gt).sum()
    union = (preds+ gt).sum() 
    eps = 1e-8

    #DICE = 2 * (intersection)/union
    dice = (2 * inter)/(union + eps)
    #IOU = intersection/ (area of non-overlap)
    iou = (inter + eps)/(union - inter + eps)
    #pixel accuracy
    acc = (preds== gt).sum()/torch.numel(preds)

    metrics = {"dice": dice, "iou": iou, "acc": acc}
        
    return metrics


    
def getRLE(img):
    """ 
    Use run-length encoding (RLE) to encode a mask image provided as numpy array.
    1: in mask, 0: background. 
    RLE is a compact way to store a mask. Return the formated RLE.

    For e.g.:
    (1 20 30 2): indicates the starting at pixel 1, the next 20 pixels have a value of 1. Then
    starting at pixel 30, the next 2 pixels (30, 31), have a value of 1.
    The pixels are numbered from top to down and then from left to right.
    """    

    #convert to a 1-dim vector 
    pixels = img.flatten()
    #print(pixels.shape)
    pixels[0] = pixels[-1] = 0
    #compute RLE
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    #print(runs[1::2], runs[:-1:2])
    runs[1::2] -= runs[:-1:2]

    #format RLE as a string
    rle = " ".join(str(i) for i in runs)
    
    return rle
            

       

