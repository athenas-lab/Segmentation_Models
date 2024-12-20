import os
import time
import copy
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms.functional as ttf
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet.unet_seg_bn import UNet 
import eval_utils as eu

""" Train a UNet model for segmentation """

class TrainUNetSeg:


    def __init__(self, cfg):

        self.cfg = cfg  
        #instantiate the Unet model
        self.model =  UNet(cfg).to(cfg.device)
        #for binary classification we use binary x-entropy loss function
        self.loss_fn = BCEWithLogitsLoss()
        #optimizer
        self.opt = Adam(self.model.parameters(), lr=cfg.init_lr)

    def trainModel(self, train_dl, val_dl):

        start_time = time.time() #start timer
        print(f"Training started at {start_time}")

        epoch_train_loss = 0
        loss_history = {"train":[], "val": []}
        
        #track the best performance
        best_val_loss = np.inf
        best_val_epoch = 0
        best_model = None

        #run the training loop
        for ep in tqdm(range(self.cfg.nepochs)):
            #activate training model
            self.model.train()
            epoch_train_loss = 0.0

            #train each batch
            for i, batch in enumerate(train_dl):
                
                #clear accumulated grads
                self.opt.zero_grad()
                
                im_name, x, y = batch["img_name"], batch["image"], batch["mask"]
                #map data to the device
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)   

                #forward prop to predict the seg map and compute loss
                preds = self.model(x) 
                #compute loss wrt GT map
                loss = self.loss_fn(preds, y)

                #print(f"Train: GT mask shape: {y.shape}, Pred shape: {preds.shape}")

                #backprop the loss and update model parameters 
                loss.backward()
                self.opt.step()

                epoch_train_loss += loss

            #compute average loss for the epoch
            avg_train_loss = epoch_train_loss /(i+1)
            # print the model  information
            print("Train::EPOCH:{}/{}: Train loss = {:.6f}".format(ep, self.cfg.nepochs-1, avg_train_loss))
            
            #validate  the model and compute validation loss 
            avg_val_loss = self.validateModel(val_dl)
            
            #track the training and validation performance
            loss_history["train"].append(avg_train_loss.cpu().detach().numpy())
            loss_history["val"].append(avg_val_loss.cpu().detach().numpy())

            #record performance if model improved
            if avg_val_loss < best_val_loss:
               best_val_loss = avg_val_loss
               best_val_epoch = ep
               best_model = copy.deepcopy(self.model)
               print("Model improved performance. Validation loss: {:.4f}".format(avg_val_loss))
               torch.save(best_model.state_dict(), os.path.join(self.cfg.model_path, "best_model.pth"))
               print(f"Saving model: epoch={ep}")
               self.savePreds(ep, val_dl)
            else:
                print("Model did not improve performance. Validation loss: {:.4f}".format(avg_val_loss))
                if (ep - best_val_epoch) >= 10:  
                    print(f"Early Stopping at epoch {ep}. Best model at ep {best_val_epoch} with val loss = {best_val_loss}")
                    break
        end_time = time.time()
        dur = (end_time - start_time)/60.0
        print(f"Training completed in {dur} minutes")
     
        #save last model
        #torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, "ep_%d.pth"%ep))
      
        eu.displayPerf(loss_history)
     
        return        

    def validateModel(self, val_dl):
        """ Evaluate model """

        with torch.no_grad():
        
            self.model.eval()            
            epoch_val_loss = 0
            #validate each batch
            for i, batch in enumerate(val_dl):
              
                im_name, x, y = batch["img_name"], batch["image"], batch["mask"]

                #map data to the device
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)   

                #forward prop to predict the seg map
                preds = self.model(x) 
                loss = self.loss_fn(preds, y)  
                epoch_val_loss += loss

        avg_val_loss = epoch_val_loss /(i+1)

        return avg_val_loss


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
                preds = torch.sigmoid(self.model(x))
                preds = (preds > self.cfg.threshold).float() #compute the binary masks

                #save results                
                torchvision.utils.save_image(preds, self.cfg.results_path+"/pred_%d_%d.jpg"%(ep, i))
                torchvision.utils.save_image(y, self.cfg.results_path+"/gt_%d_%d.jpg"%(ep, i))

                return        



