import os
import sys
import time
import copy
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms.functional as ttf
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

sys.path.append("../")
import common.viz_seg_data as viz


""" Train a fully convolutional model (Unet or FPN) for segmentation """

class TrainFullyConvNetSeg:
    """ 
    Train a fully conv net (FCN) for semantic segmentation task.
    Currently supported FCN: 
    - Unet (with and without batchnorm (original network))
    - Feature Pyramid Net (FPN) with Resnet backbone
    """

    def __init__(self, cfg, labeler):

        self.cfg = cfg  
        self.mask_labeler = labeler
        #instantiate the segmentation model
        self.model = self.getModel()
        self.loss_fn = self.getLoss()
        #optimizer
        self.opt = Adam(self.model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay)
        #decay LR by lr_decay every step_size number of epochs 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size = 10, gamma = cfg.lr_decay)
        print(self.lr_scheduler.get_last_lr()[0])

    def getModel(self):
        """ Supported models: Unet (with or without batchnorm), FPN """

        if self.cfg.model_name.startswith("unet"):
           from models.unet.unet_seg_bn import UNet

           model =  UNet(self.cfg).to(self.cfg.device)

        elif self.cfg.model_name.startswith("fpn"):
           from models.fpn.fpn_resnet import SegmentFPN

           model =  SegmentFPN(self.cfg).to(self.cfg.device)

        return model

    def getLoss(self):
        """ Supported loss functions: Binary or categorical cross entropy """
        if self.cfg.loss_fn == "bce":
           #for binary classification we use binary x-entropy loss function
           loss_fn = nn.BCEWithLogitsLoss()

        elif self.cfg.loss_fn == "ce": 
           #categorical classification 
           loss_fn = nn.CrossEntropyLoss(ignore_index=self.cfg.ignore_index)
           
        return loss_fn

    def adjustLearningRate(self, decay=0.01):
        """ Gradually lower the learning rate as training progresses """

        lr_decay = decay
        if self.cfg.lr_decay is not None:
           lr_decay = self.cfg.lr_decay
        for param_group in self.opt.param_groups:
            param_group["lr"] -=  lr_decay * param_group["lr"] 
        
        #self.lr_scheduler.step()
        print("Learning rate adjusted to %f"%param_group["lr"]) #{self.lr_scheduler.get_last_lr()[0]}")
        return

    
    def trainModel(self, train_dl, val_dl):

        start_time = time.time() #start timer
        print(f"Training started at {start_time}")

        epoch_train_loss = 0
        loss_history = {"train":[], "val": []} #cross-entropy loss
        mIOU_history = {"train":[], "val": []} #mean IOU 
        dice_history = {"train":[], "val": []} #mean DICE
        pix_acc_history = {"train":[], "val": []} #pixel accuracy

        #track the best performance
        best_val_loss = np.inf
        best_val_epoch = 0
        best_model = None

        #run the training loop
        for ep in tqdm(range(self.cfg.nepochs)):
            #activate training model
            self.model.train()
            epoch_train_loss = 0.0
            #running metrics initialization
            train_inter_per_epoch = 0
            train_union_per_epoch = 0
            train_correct_per_epoch = 0
            train_labeled_per_epoch = 0


            count = 0
            #train each batch
            for i, batch in enumerate(train_dl):
                count += 1
                #clear accumulated grads
                self.opt.zero_grad()
                
                im_name, x, y = batch["img_name"], batch["image"], batch["mask"]
                #map data to the device
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)   
                #print(x.shape, y.shape)
                #forward prop to predict the seg map and compute loss
                model_out = self.model(x)
                #print(model_out, y, model_out.shape, y.shape)

                #compute loss wrt GT map            
                loss = self.loss_fn(model_out, y)
                epoch_train_loss += loss.item() #cpu().detach().numpy()
                #print(f"Train: GT mask shape: {y.shape}, Pred shape: {model_out.shape}")

                # compute segmentation metrics for training batch
                model_out = torch.sigmoid(model_out)
             
                if self.cfg.loss_fn == "bce":
                   preds = (model_out > self.cfg.threshold)*1 #compute the binary masks
                elif self.cfg.loss_fn == "ce":
                   preds = torch.argmax(model_out, dim=1) #class ids with highest likelihood
                
                #print(y, preds, preds.min(), preds.max(), y.min(), y.max())


                #backprop the loss and update model parameters 
                loss.backward()
                self.opt.step()

               
            #-----------------training metrics ------------------

            #compute average loss for the epoch
            avg_train_loss = epoch_train_loss /count

  
            # print the model  information
            print("Train::EPOCH:{}/{}: Train loss = {:.6f}".format(ep, self.cfg.nepochs-1, avg_train_loss)) 

            #track the training performance
            loss_history["train"].append(avg_train_loss)
 
            #----------------------------------------------------

            if ep % self.cfg.epoch_interval_val == 0: 
                #validate  the model and compute validation loss 
                val_metrics = self.validateModel(val_dl)
                avg_val_loss = val_metrics["avg_loss"]
                
                #track the validation performance
                loss_history["val"].append(avg_val_loss)


                #record performance if model improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_epoch = ep
                    best_model = copy.deepcopy(self.model)
                    print("Model improved performance. Validation loss: {:.4f}".format(avg_val_loss))

                    torch.save(best_model.state_dict(), os.path.join(self.cfg.model_path, "best_model.pth"))
                    print(f"Saving model: epoch={ep}")
                    if self.cfg.loss_fn == "bce":
                       self.savePreds(ep, val_dl)
                else:
                    print("Model did not improve performance. Validation loss: {:.4f}".format(avg_val_loss))

                    self.adjustLearningRate()
                    if (ep - best_val_epoch) >= self.cfg.early_stop_epochs:  
                        print(f"Early Stopping at epoch {ep}. Best model at ep {best_val_epoch} with val loss = {best_val_loss}")
                        break
        end_time = time.time()
        dur = (end_time - start_time)/60.0
        print(f"Training completed in {dur} minutes")
     
        #save last model
        #torch.save(best_model.state_dict(), os.path.join(self.cfg.model_path, "ep_%d.pth"%ep))
      
        displayPerf(loss_history, "Avg Loss")

     
        return        

    def validateModel(self, val_dl):
        """ Evaluate model """

        with torch.no_grad():
        
            self.model.eval()            
            epoch_val_loss = 0

            #running metrics initialization
            val_inter_per_epoch = 0
            val_union_per_epoch = 0
            val_correct_per_epoch = 0
            val_labeled_per_epoch = 0

            logits = None
            labels = None

            count = 0
            #validate each batch
            for i, batch in enumerate(val_dl):
                count += 1
                im_name, x, y = batch["img_name"], batch["image"], batch["mask"]

                #map data to the device
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)   

                #forward prop to predict the seg map
                model_out = self.model(x)
                #if self.cfg.loss_fn  == "bce":
                #for BCELogitsLoss, the loss fn computes sigmoid inherently  
                loss = self.loss_fn(model_out, y)  
                model_out = torch.sigmoid(model_out)
                #elif self.cfg.loss_fn  == "ce":
                #    #for CrossEntropyLoss, we pass the unnormalized logits and ground-truth, which  is in [0, C]
                #    loss = self.loss_fn(model_out, y)  
                epoch_val_loss += loss.item()

                logits = model_out if logits is None else torch.cat((logits, model_out), dim=0) 
                labels = y if labels is None else torch.cat((labels, y), dim=0)
             
                if self.cfg.loss_fn == "bce":
                   preds = (model_out > self.cfg.threshold)*1 #compute the binary masks
                elif self.cfg.loss_fn == "ce":
                   preds = torch.argmax(model_out, dim=1) #class ids with highest likelihood
       

        avg_val_loss = epoch_val_loss /count

        val_metrics={ "avg_loss": avg_val_loss,
                 }


        preds_npy, labels_npy = getPredLabels(self.cfg, [logits, labels])
        computeMetricsAndViz(self.cfg, self.mask_labeler, preds_npy, labels_npy)

        return val_metrics

  

    def savePreds(self, ep, val_dl):
        """ Evaluate model and save predicted masks as images """

        with torch.no_grad():
        
            self.model.eval()            

            #validate each batch
            for i, batch in enumerate(val_dl):
             
                im_name, x, y = batch["img_name"], batch["image"], batch["mask"]
                #map data to the device
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)   

                #forward prop to predict the seg map
                model_out = self.model(x)
                preds = model_out.detach().cpu()
                if self.cfg.loss_fn == "bce":
                   preds = torch.sigmoid(preds)
                   preds = (preds > self.cfg.threshold).float() #compute the binary masks
                   #save target and predicted masks           
                   torchvision.utils.save_image(preds, self.cfg.results_path+"/pred_%d_%d.jpg"%(ep, i))
                   torchvision.utils.save_image(y, self.cfg.results_path+"/gt_%d_%d.jpg"%(ep, i))

                return        

def getPredLabels(cfg, logit_labels):

    with torch.no_grad():
            logits, labels = logit_labels
            #print(logits, labels, logits.shape, labels.shape)
            #upsample the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits,
                size=labels[0].shape[-2:], #height, width
                mode="bilinear",
                align_corners=False #True only works with linear|bilinear|bicubic
            )

            pred_labels = logits_tensor.detach().cpu().numpy()

            if cfg.loss_fn == "bce":
               pred_labels = (pred_labels > cfg.threshold)*1 #compute the binary masks
            elif cfg.loss_fn == "ce":
               pred_labels = pred_labels.argmax(axis=1) #class ids with highest likelihood

            labels = labels.cpu().numpy()

    return pred_labels, labels

def computeMetricsAndViz(cfg, labeler, preds, labels, show=False):

        """ 
        Given the ground-truth and prediction, 
        compute the metrics (IOU and accuracy) of the segmentation model.
        """

        metric = evaluate.load("mean_iou")
        id2lab = labeler.id2label
        #print(id2lab)
        labels = labels.squeeze()
        preds= preds.squeeze()   
      
        #print(pred_labels, labels, pred_labels.shape, labels.shape)
        metrics = metric._compute(  #using _compute() instead of compute() for speedup
                predictions=preds,
                references=labels,
                num_labels=len(id2lab),
                ignore_index=cfg.ignore_index,
                reduce_labels=cfg.reduce_labels
            )
             
        #get the accuracy and IOU for each class
        per_category_acc = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        #per_category_dice = metrics.pop("per_category_dice").tolist()
        metrics.update({f"acc_{labeler.id2label[i]}": v for i, v in enumerate(per_category_acc)})
        metrics.update({f"iou_{labeler.id2label[i]}": v for i, v in enumerate(per_category_iou)})
        print("Test dataset:", metrics) 

        if show:   
            #visualize predicted and ground-truth masks
            print(preds.shape, labels.shape, "range:", np.min(preds), np.max(preds))
            #map class labels to RGB values
            pred_labels = labeler.classToRGBMask(preds)
            labels = labeler.classToRGBMask(labels)
            print(pred_labels.shape, labels.shape)
            #viz.compareSingleGTPredMask(pred_labels[0,...], labels[0,...])
            viz.showGTPredMasks(labels, pred_labels, 5)
       
        return metrics


def evaluateModel(cfg, model, test_dl, mask_labeler):
    """ Evaluate a trained model on the test dataset. Compute metrics and visualize """

    if model is None: #no model is provided
        # load our pretrained model from disk and flash it to the current device
        print("Loading the trained model for prediction...")
        if cfg.model_name.startswith("unet"):
           from models.unet.unet_seg_bn import UNet

           model =  UNet(cfg).to(cfg.device)

        elif cfg.model_name.startswith("fpn"):
           from models.fpn.fpn_resnet import SegmentFPN

           model =  SegmentFPN(cfg).to(cfg.device)

        model.load_state_dict(torch.load(os.path.join(cfg.model_path, "best_model.pth"), weights_only=True))
     
    model.eval()            
    logits = None
    labels = None
    imgs_all = None
    preds_all = None
    gt_exists = False

    with torch.no_grad():
        #validate each batch
        for i, batch in enumerate(test_dl):
                
            if "mask" in batch.keys(): #GT masks exist
                img_names, x, y = batch["img_name"], batch["image"], batch["mask"]
                #GT mask exists 
                gt_exists = True 
                #map data to the device
                x, y = x.to(cfg.device), y.to(cfg.device)
                labels = y if labels is None else torch.cat((labels, y), dim=0)

            else: 
                #GT mask does not exist 
                img_names,  x = batch["img_name"], batch["image"]
                #map data to the device
                x = x.to(cfg.device)
                y = None

            #forward prop to predict the seg map
            model_out = model(x)
            model_out = torch.sigmoid(model_out)

            logits = model_out if logits is None else torch.cat((logits, model_out), dim=0) 
            if imgs_all is None:  #accumulate for visualization
                imgs_all = x.cpu().numpy() #map images back to cpu to visualize
            else:
                imgs_all = np.vstack((imgs_all, x.cpu().numpy()))    

            if gt_exists: 

                if cfg.loss_fn == "bce":
                    preds = (model_out > cfg.threshold)*1 #compute the binary masks
                elif cfg.loss_fn == "ce":
                    preds = torch.argmax(model_out, dim=1) #class ids with highest likelihood
                preds =  preds.detach().cpu()  
                preds_all =  preds if preds_all is None else torch.cat((preds_all, preds), dim=0) 

    if gt_exists:   #if GT masks exist, compute metrics and visualize the predicted and ground-truth masks    
        
        #print(logits, labels, preds_all, logits.shape, labels.shape, preds_all.shape)
        preds_npy, labels_npy = getPredLabels(cfg, [logits, labels])
        metrics = computeMetricsAndViz(cfg, mask_labeler, preds_npy, labels_npy, show=True)
        #metrics = computeMetricsAndViz(cfg, mask_labeler, [logits.squeeze(), labels.squeeze()], show=True)
    else: 
        #no metric computation when there is no ground-truth labels. Only visualization.

        logits_tensor = nn.functional.interpolate(
                logits,
                size=imgs[0].shape[-2:], #height, width
                mode="bilinear",
                align_corners=False #True only works with linear|bilinear|bicubic
            )

        pred_labels = logits_tensor.detach().cpu().numpy()

        if cfg.loss_fn == "bce":
            pred_labels = (pred_labels > cfg.threshold)*1 #compute the binary masks
        elif cfg.loss_fn == "ce":
            pred_labels = pred_labels.argmax(axis=1) #class ids with highest likelihood
        
        #visualize    
        imgs_all = np.transpose(imgs_all, (0, 2, 3, 1))
        #visualize predicted masks
        print(imgs_all.shape, pred_labels.shape, "range:", np.min(pred_labels), np.max(pred_labels))
        #map class labels to RGB values
        pred_labels = mask_labeler.classToRGBMask(pred_labels)
        print(pred_labels.shape)
        viz.showImageWithPredMasks(imgs, pred_labels, 5)
        
    
    return

def displayPerf(hist, title):

    # plot the loss values over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Value")
    plt.legend(loc="lower left")
    plt.show()

    return
