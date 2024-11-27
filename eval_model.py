import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as ttf

from models.unet.unet_seg_bn import UNet #Unet with optional batchnorm
import eval_utils as eu

""" Evaluate pre-trained segmentation models on different datasets """



def evalModel(cfg, test_dl, model=None, rle=False):
    """ Predict masks for test dataset """

    if model is None: #no model is provided
        # load our pretrained model from disk and flash it to the current device
        print("Loading the trained model...")
        model =  UNet(cfg).to(cfg.device)
        model.load_state_dict(torch.load(os.path.join(cfg.model_path, "best_model.pth"), weights_only=True))

    rle_preds = []    
    all_preds = None
    all_gt = None
    gt_exists = False
    for i, batch in enumerate(test_dl):
        if "mask" in batch.keys():
            img_names, x, y = batch["img_name"], batch["image"], batch["mask"]
            #GT mask exists 
            gt_exists = True 
            #map data to the device
            x = x.to(cfg.device)
        else: 
            #GT mask does not exist 
            img_names,  x = batch["img_name"], batch["image"]
            #map data to the device
            x = x.to(cfg.device)
            y = None

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds =  (preds > cfg.threshold).float() #compute the binary masks
       
            #save visual results               
            if i ==0:
               torchvision.utils.save_image(preds, cfg.results_path+"/test_pred_%d.jpg"%(i))
               torchvision.utils.save_image(x, cfg.results_path+"/img_%d.jpg"%(i))

        if gt_exists: #compute quantitative metrics
            if all_preds is None:
               all_preds = preds
               all_gt = y
            else:
               all_preds = torch.cat((all_preds, preds), dim=0)    
               all_gt = torch.cat((all_gt, y), dim=0)    
            #print(all_preds.shape, all_gt.shape)   

        if rle: #compute run-length encoding for masks (for kaggle submission)   
            preds = ttf.resize(
                preds, size=(cfg.orig_img_h, cfg.orig_img_w), #resize to original size
                    interpolation = ttf.InterpolationMode.NEAREST)

            for j in range(len(img_names)):
                #encode the mask using run-length encoding for compactness
                encoding = eu.getRLE(preds[j].squeeze().cpu())
                rle_preds.append((img_names[j], encoding))
        
    if all_preds is not None:

        all_preds = all_preds.cpu().squeeze()
        all_gt = all_gt.squeeze()
        metrics = eu.computeSegMetricsTorch(all_gt, all_preds, cfg.num_classes)
        for k, v in metrics.items():
            print(f"{k} = {v}")
      


    #for kaggle submission 
    if len(rle_preds) != 0:
       sub = pd.DataFrame(rle_preds)
       sub.columns = ["img", "rle_mask"]
       sub.to_csv(os.path.join(cfg.results_path, "submission.csv"), index=False)  

    return


