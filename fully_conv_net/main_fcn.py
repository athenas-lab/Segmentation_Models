import sys
sys.path.append("../")
import torch

import common.config as conf
import common.data_utils  as dut
import common.viz_seg_data as viz
import train_fcn as tm

""" Train and evaluate fully convolutional models for semantic segmentation using different datasets """

def trainModel(cfig, labeler):

   train_dl, val_dl = data.loadTrainingData(cfig, labeler)
       
   if cfig.debug:
      #if cfig.num_classes > 1:
      viz.showMultiClassSegData(val_dl, labeler)
      #else:   
      #   viz.showSegData(val_dl)
      exit()

   trainer = tm.TrainFullyConvNetSeg(cfig, labeler)
   model = trainer.trainModel(train_dl, val_dl)

   return model, val_dl


def testModel(cfig, model=None, labeler=None):
    """ Predict masks for test dataset """

    if cfig.data_name == "camvid": #camvid dataset has test split with ground-truth
       test_dl = data.loadTestData(cfig, labeler, gt_mask=True)
    else: #for datasets where there is no test split, use val split for testing
       _, test_dl = data.loadTrainingData(cfig, labeler) 
    tm.evaluateModel(cfig, model, test_dl, labeler) 
    return


if __name__ == "__main__":
   
   data_name = "carvana"
   model_name = "fpn" #["fpn", "unet_bn", "unet"]
   loss_fn = "bce"  #["bce", "ce"]
   mode = "predict" #["train", "predict"]
   cfg = conf.Config(data_name, model_name, loss_fn, mode=mode)
   mask_labeler = dut.SegMapClassLabeler(cfg.label_info)

   if data_name == "diva":
      import data.diva_data  as data
   elif data_name == "carvana":
      import data.carvana_data  as data
   elif data_name == "salt":
      import data.salt_data  as data   
   elif data_name == "camvid":
      import data.camvid_data  as data   

   if mode == "train":
      model, val_dl = trainModel(cfg, mask_labeler)
      #valModel(cfig, val_dl, model, mask_labeler)
   else:
     trained_model = None #None = load saved model from file
     #test a pretrained model or pass a model for testing.
     testModel(cfg, trained_model, mask_labeler)


