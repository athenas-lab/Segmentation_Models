import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms.functional as ttf
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
from torch.optim import Adam
import evaluate

from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation,
    TrainingArguments, Trainer
   )


sys.path.append("../")
import common.viz_seg_data as viz


""" Finetune a Segformer model for semantic segmentation """

class FineTuneSegformer:
    """ Fine-tune a pretrained segformer model """

    def __init__(self, cfg,  img_proc, mask_labeler, pretrained_model):

        self.cfg = cfg  
        self.mask_labeler = mask_labeler
        self.pretrained_model = pretrained_model
        self.img_proc =  img_proc
        print(mask_labeler.num_classes)
        #instantiate the Segformer model
        self.model =  SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model, 
                return_dict=False, 
                num_labels=mask_labeler.num_classes,
                id2label=mask_labeler.id2label,
                label2id=mask_labeler.label2id,
                ignore_mismatched_sizes=True
            )

      
        self.training_args = TrainingArguments(
                output_dir = cfg.results_path,
                learning_rate=cfg.init_lr,
                num_train_epochs=cfg.nepochs,
                per_device_train_batch_size=cfg.batch_size,
                per_device_eval_batch_size=cfg.batch_size,
                save_total_limit=3,
                save_strategy="epoch", #"steps",
                eval_strategy="epoch",
                logging_strategy="epoch",
                #save_steps=20,
                #eval_steps=1,
                logging_steps=1,
                eval_accumulation_steps=5,
                load_best_model_at_end=True,
                push_to_hub=False,
                report_to = "none"  
            )
        print(self.training_args.learning_rate)
        self.save_model_path = cfg.model_path    

    def finetuneModel(self, train_ds, val_ds):
        """ Fine-tune the segformer model """

        trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_ds, #Dataset
                eval_dataset=val_ds,   #Dataset
                compute_metrics=self.computeMetrics
        )
     
        train_result = trainer.train()
        trainer.save_model(self.save_model_path)


        # compute train results
        metrics = train_result.metrics
        # save train results
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # compute evaluation results on train dataset
        metrics = trainer.evaluate(eval_dataset=train_ds)
        # save evaluation results
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # compute evaluation results on val dataset
        metrics = trainer.evaluate(eval_dataset=val_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        return self.model


    def  evalModel(self, test_ds):
        """ Evaluate the segformer model on the test data with ground-truth masks """

    
        trainer = Trainer(
                    model=self.model,
                    args=self.training_args,
                    eval_dataset=test_ds,   #Dataset
                    compute_metrics=self.computeMetricsAndViz
            )

        # compute evaluation results on test dataset
        metrics = trainer.evaluate(eval_dataset=test_ds)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        return 


    def computeMetrics(self, eval_pred):

        """ 
        Given the ground-truth and prediction, 
        compute the metrics (IOU and accuracy) of the segmentation model.
        """

        metric = evaluate.load("mean_iou")
        id2lab = self.mask_labeler.id2label
        
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            

            #scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:], #height, width
                mode="bilinear",
                align_corners=False
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric._compute(  #using _compute() instead of compute() for speedup
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2lab),
                ignore_index=self.cfg.ignore_index,   #default value of ignore_index for the loss fn in SegformerForSemanticSegmentation
                reduce_labels=self.img_proc.do_reduce_labels
            )
           
            #get the accuracy and IOU for each class
            per_category_acc = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()
          
            metrics.update({f"acc_{id2lab[i]}": v for i, v in enumerate(per_category_acc)})
            metrics.update({f"iou_{id2lab[i]}": v for i, v in enumerate(per_category_iou)})
            print(metrics)
            #print(f"Mean IOU={metrics["mean_iou"]}, Mean accuracy={metrics["mean_accuracy"]}, Overall accuracy={metrics["overall_accuracy"]}")
            return metrics


    def computeMetricsAndViz(self, eval_pred):
        """ Predict seg masks for test dataset and display masks """


        metric = evaluate.load("mean_iou")
        id2lab = self.mask_labeler.id2label

        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            
            #upsample the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:], #height, width
                mode="bilinear",
                #align_corners=False #only works with linear|bilinear|bicubic
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric._compute(  #using _compute() instead of compute() for speedup
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2lab),
                ignore_index=self.cfg.ignore_index,
                reduce_labels=self.img_proc.do_reduce_labels
            )
           
            #get the accuracy and IOU for each class
            per_category_acc = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"acc_{id2lab[i]}": v for i, v in enumerate(per_category_acc)})
            metrics.update({f"iou_{id2lab[i]}": v for i, v in enumerate(per_category_iou)})

            print(f"Test dataset: Mean IOU={metrics["mean_iou"]}, Mean accuracy={metrics["mean_accuracy"]}, Overall accuracy={metrics["overall_accuracy"]}")


            #visualize predicted and ground-truth masks
            print(pred_labels.shape, labels.shape, "range:", np.min(pred_labels), np.max(pred_labels))
            #map class labels to RGB values
            pred_labels = self.mask_labeler.classToRGBMask(pred_labels)
            labels = self.mask_labeler.classToRGBMask(labels)
            print(pred_labels.shape, labels.shape)
            #viz.compareSingleGTPredMask(pred_labels[0,...], labels[0,...])
            viz.showGTPredMasks(labels, pred_labels, 5)
       
        return metrics
            

    
def predictMasksAndViz(cfg, model_ckpt, mask_labeler, test_dl):
    """ Predict seg masks for test dataset and display masks """

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #instantiate the Segformer model from pretrained checkpt
    model = SegformerForSemanticSegmentation.from_pretrained(
                model_ckpt, 
                return_dict=False, 
                num_labels=mask_labeler.num_classes,
                id2label=mask_labeler.id2label,
                label2id=mask_labeler.label2id,
                ignore_mismatched_sizes=True
            )
    model.to(device)        
    model.eval()

    with torch.no_grad():
        imgs_all = None
        pred_labels = None
        labels_all = None
        #predict labels for the full dataset
        for i, batch in enumerate(tqdm(test_dl)):
            imgs = batch["pixel_values"]
            imgs = imgs.to(device)
    
            #forward pass to get the model predictions
            logits = model(imgs)[0]
            #print(logits, logits.shape)

            #Segformer outputs logits in the shape (batch, num_labels, H/4, W/4). 
            #So need to upscale the mask to size of the image   
            upsampled_logits = torch.nn.functional.interpolate(
                logits, 
                size=imgs[0].shape[-2:], 
                mode="bilinear", 
                #align_corners=False
                )

            #get predictions from logits 
            preds = upsampled_logits.argmax(dim=1).detach().cpu().numpy()
            #print(preds.shape)
            
            #if i == 1:
            #   print(imgs[0], imgs_all[0])
            #   print(logits[0], logits_all[0])
            if imgs_all is None:  #accumulate for optional visualization
                imgs_all = imgs.cpu().numpy() #map images back to cpu to visualize
            else:
                imgs_all = np.vstack((imgs_all, imgs.cpu().numpy()))    

            if "labels" in batch.keys(): #GT masks exist
                labels = batch["labels"]  
                #accumulate for computing metrics and optional visualization
                if labels_all is None:
                   labels_all = labels.cpu().numpy()
                else:
                    labels_all = np.vstack((labels_all, labels.cpu().numpy()))             

            if pred_labels is None:
                pred_labels = preds
            else:
                pred_labels = np.vstack((pred_labels, preds)) 
            #print(pred_labels.shape, imgs_all.shape)    

       
        if labels_all is not None:
            #compute metrics if ground-truth masks are available
            computeMetrics(labels_all, pred_labels, cfg, mask_labeler)
        
        #visualize    
        imgs = np.transpose(imgs_all, (0, 2, 3, 1))
        #visualize predicted masks
        print(imgs.shape, pred_labels.shape, "range:", np.min(pred_labels), np.max(pred_labels))
        #map class labels to RGB values
        pred_labels = mask_labeler.classToRGBMask(pred_labels)
        print(pred_labels.shape)
        if labels_all is not None:
            labels = mask_labeler.classToRGBMask(labels_all)
            print(pred_labels.shape, labels.shape)         
            viz.showImagesWithPredMasks(imgs, pred_labels, 5, labels)
            #viz.showImagesWithPredMasks(imgs[-10:,...], pred_labels[-10:,...], 5, labels[-10:,...])
        else:    #no gt masks
            viz.showImagesWithPredMasks(imgs, pred_labels, 5)
        #viz.showGTPredMasks(labels, pred_labels, 5)
            
            
        return


def computeMetrics(labels, pred_labels, cfg, mask_labeler):
        """ Predict seg masks for test dataset and display masks """

        metric = evaluate.load("mean_iou")
        id2lab = mask_labeler.id2label

          
        metrics = metric._compute(  #using _compute() instead of compute() for speedup
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2lab),
            ignore_index=cfg.ignore_index,
            reduce_labels=cfg.reduce_labels
        )
           
        #get the accuracy and IOU for each class
        per_category_acc = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"acc_{id2lab[i]}": v for i, v in enumerate(per_category_acc)})
        metrics.update({f"iou_{id2lab[i]}": v for i, v in enumerate(per_category_iou)})

        for k, v in metrics.items():
            print(f"{k}: {v}")

        return metrics    
