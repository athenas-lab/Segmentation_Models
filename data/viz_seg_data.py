import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2


from . import data_utils as dut



""" 
Visualization utilities for binary and multi-class image segmentation.
Modified from   https://github.com/CherifiImene/buildings_and_road_segmentation
"""

def showImgSegMask(img, seg):
    """ Visualize the image and segmentation map """

    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #seg = cv2.imread(seg_path)
    #seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    images = [img, seg] 
    for i, data in enumerate(images):   
        plt.subplot(1, 2, i+1)
        plt.imshow(data)
    plt.show()

    return

def compareSingleGTPredMask(gt_mask, pred_mask, save_file):
    """ Show a single ground-truth segmentation mask with predicted mask """

    img_mask = pred_mask.astype(np.uint8)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(gt_mask)
    axs[1].imshow(img_mask)

    plt.show()

    if save_file is None:
       fig.savefig(save_file)

    return 


def showOverlayMask(img, seg):
    """ Overlay the segmentation map on the image and display the blended image """

    alpha = 0.6 #degree of transparency
    beta = 1-alpha
    gamma = 0 #scalar added to each sum

    new_img = np.array(img) #create a copy for overlay
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    new_seg = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)

    blend = cv2.addWeighted(new_seg, alpha, new_img, beta, gamma, new_img)
    plt.imshow(blend)
    plt.show()
    return blend

def showSegData(dl):
    """ Preview the binary segmentation data (image, mask) after loading """

    #batch = next(iter(dl))
    #print("Batch size = %d"%len(batch))
    for i, b in enumerate(dl):
        assert("mask" in b.keys())
        fn, img, msk = b["img_name"], b["image"], b["mask"]
        print(img.shape, msk.shape)
        x = dut.tensor2Img(img[0])
        #x.show()
        #mask = msk[0].unsqueeze(0)
        print(f"img shape: {img[0].shape}, mask shape:{msk[0].shape}, image name:{fn[0]}")
        x = dut.tensor2Img(msk[0])
        #x.show()
        visImgSegMap(x, y)

        return


def showMultiClassSegData(dl, ml):
    """ Preview the multiclass segmentation  data (image, seg map) after loading """

    #batch = next(iter(dl))
    #print("Batch size = %d"%len(batch))
    for i, b in enumerate(dl):
        assert("mask" in b.keys())
        fn, img, msk = b["img_name"], b["image"], b["mask"]
        print(img.shape, msk.shape)

        #unnormalize the images if they were normalized during data processing
        #x = np.transpose(img[0].numpy(), (1, 2, 0)) #CHW --> HWC
        #x = x * std + mean
        #x = np.array(x, dtype=np.float32)
        x = dut.tensor2ImgMeanStd(img[0])
        x = np.array(x, dtype=np.float32)
        print(f"img shape: {img[0].shape}, mask shape:{msk[0].shape}, image name:{fn[0]}")
        #y = dut.tensor2SegMap(msk[0])
        #y = ml.labelsToSegMap(msk[0])
        y = msk[0].numpy()
        showImgSegMap(x, y)
        return



def showImageWithRGBMask(imgs, masks, labeler, nsamples, random= False, overlay=False):
    """ 
    Display images with the corresponding RGB segmentation 
    after converting grayscale labels to RGB mask.
    
    overlay: True:: overlay mask on image
             False: show image and mask separately
    """

    if overlay:
        fig, axes = plt.subplots(1, nsamples, figsize=(8*nsamples, 4))
    else: #image and mask in separate rows
        fig, axes = plt.subplots(2, nsamples, figsize=(8*nsamples, 4))

    #print(len(imgs), len(masks), imgs[0].shape, masks[0].shape, torch.min(masks[0]), torch.max(masks[0]))
    nsamples = min(nsamples, len(imgs))

    if random:
        samples = np.random.choice(len(imgs), nsamples, replace=False)
    else:
        samples = range(nsamples)          

    for idx, sample in enumerate(samples):
        img = imgs[idx]
        mask = masks[idx] 
        #print(img.shape, mask.shape)
        #img = np.transpose(img, (1, 2, 0)) #(h, w, c)
        rgb = labeler.classToRGBMask(mask) 
        if overlay:
            i1 = axes[idx].imshow(img)
            i2 = axes[idx].imshow(rgb, alpha=0.7)  
        else: #show image and mask separately
            i1 = axes[0, idx].imshow(img)
            axes[0, idx].set_title("Image")
            i2 = axes[1, idx].imshow(rgb)  
            axes[1, idx].set_title("Mask")
    plt.show()
    return


def showImagesWithPredMasks(images, preds, nsamples, gt_masks=None, overlay=False):
    """ 
      Display GT and predicted masks for n samples.
      Assumes pixels in gt_mask and preds have been converted to RGB labels
    """

    cnt = min (nsamples, preds.shape[0])
    if overlay == False: #display image and mask separately
        if gt_masks is None:
            f, axarr = plt.subplots(cnt, 2, figsize=(8,24))
        else:
            f, axarr = plt.subplots(cnt, 3, figsize=(8,24))   
        for i in range(cnt):
            #images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) 
            axarr[i,0].imshow(images[i,...])
            axarr[i,1].imshow(preds[i,...])
            if gt_masks is not None:
               #gt_masks[i, ...] = cv2.cvtColor(gt_masks[i, ...], cv2.COLOR_BGR2RGB) 
               axarr[i,2].imshow(gt_masks[i,...])
            if i == 0: 
               axarr[i,0].set_title("Image") 
               axarr[i,1].set_title("Predicted Mask")
               if gt_masks is not None:
                 axarr[i,2].set_title("Ground Truth Mask")

    else: #overlay mask on image with groubd-truth mask in a separate column, if present.
        if gt_masks is None:
            f, axarr = plt.subplots(cnt, 1, figsize=(8,24))
        else: 
            f, axarr = plt.subplots(cnt, 2, figsize=(8,24))   
        for i in range(cnt):
            axarr[i, 0].imshow(images[i, ...])  
            axarr[i, 0].imshow(preds[i,...], alpha=0.5) 
            if gt_masks is not None:
               axarr[i,1].imshow(gt_masks[i,...])
            if i == 0:
               axarr[i,0].set_title("Image with Predicted Mask")
               if gt_masks is not None:
                 axarr[i,1].set_title("Ground Truth Mask")

    plt.show()        
    
    return


def showGTPredMasks(gt_mask, preds, nsamples, overlay=False):
    """ 
      Display GT and predicted masks for n samples.
      Assumes pixels in gt_mask and preds have been converted to RGB labels
    """

    cnt = min (nsamples, preds.shape[0])
    if overlay == False: #display image and mask separately
        f, axarr = plt.subplots(cnt, 2, figsize=(8,24))
        for i in range(cnt):
                
            axarr[i,0].imshow(preds[i,...])
            axarr[i,1].imshow(gt_mask[i,...])
            if i == 0:
              axarr[i,0].set_title("Model Prediction")
              axarr[i,1].set_title("Ground Truth")

    else: #overlay mask on image
        f, axarr = plt.subplots(1, cnt, figsize=(24, 8))
        #images = np.transpose(images, (0, 2, 3, 1))
        for i in range(cnt):
            axarr[i].imshow(gt_mask[i, ...])  
            axarr[i].imshow(preds[i,...], alpha=0.5) 
    plt.show()        
    
    return


def plotClassDistribution(id2label, summary): 
    # Plots classes distribution

    plt.figure(figsize=(18,10))
    sns.barplot(y=summary, x= list(id2label))
    plt.axes().set_xticklabels(list(id2label), rotation=90)
    plt.show()

    return

def computeClassWeights(total, class_counts):
    # computes the weight of each class

    weights = []
    for class_count in class_counts:
        weights.append(total/class_count)
    return weights
