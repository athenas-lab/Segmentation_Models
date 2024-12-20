import numpy as np

import sys
sys.path.append("../")
import config as conf
import data.camvid_segform_data as data
import data.data_utils  as dut
import data.viz_seg_data  as viz
import finetune_segform as fts

""" 
Fine-tune Segformer model from HuggingFace using CamVid dataset 
to identify and locate multi-class objects in an urban environment.
"""

def getPretrainedModel(dn):
        """ Get pretrained segformer model id """
      
        #Pre-trained models
        if dn == "camvid":
           MODEL_CHECKPOINT = 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024'
        else:
           MODEL_CHECKPOINT = 'nvidia/mit-b4'

        # MODEL_CHECKPOINT = 'nvidia/mit-b0'
        # MODEL_CHECKPOINT = 'nvidia/mit-b1'
        # MODEL_CHECKPOINT = 'nvidia/mit-b2'
        # MODEL_CHECKPOINT = 'nvidia/mit-b3'
       

        return MODEL_CHECKPOINT

def trainModel(dn, mn):

    cfg = conf.Config(dn, model_name=mn, mode="train")

    model_ckpt = getPretrainedModel(dn)
    labeler = dut.SegMapClassLabeler(cfg.label_info)
    feat_ext = dut.getSegformerFeatureExtractor(cfg, model_ckpt, resize=True)
    train_dl, val_dl, train_ds, val_ds = data.loadTrainingData(cfg, labeler, feat_ext)

    if cfg.debug:
        batch = next(iter (val_dl))
        imgs = batch["pixel_values"] #(c, h, w)
        masks = batch["labels"] #(h, w, c)
        print(imgs.shape, masks.shape)
        imgs = np.transpose(imgs, (0, 2, 3, 1)) #(h, w, c)
        viz.showImageWithRGBMask(imgs, masks, labeler, nsamples=5, random=True, overlay=False)
        exit()
        
        
    trainer = fts.FineTuneSegformer(cfg, feat_ext, labeler, model_ckpt)
    model = trainer.finetuneModel(train_ds, val_ds)

    return model, val_ds


def testModel(dn, mn, model_ckpt=None):

    cfg = conf.Config(dn, model_name=mn, mode="predict")
    labeler = dut.SegMapClassLabeler(cfg.label_info)
    feat_ext = dut.getSegformerFeatureExtractor(cfg, getPretrainedModel(dn), resize=True)
    test_dl = data.loadTestData(cfg, labeler, feat_ext)

    #test using model finetuned with camvid test data with GT masks
    #tuner = fts.FineTuneSegformer(cfg, feat_ext, labeler, model_ckpt) 
    #tuner.evalModel(test_ds)

    #evaluate the model and visualize the results
    fts.predictMasksAndViz(cfg, model_ckpt, labeler, test_dl)

   
    return

if __name__ == "__main__":
   
    data_name = "camvid" #["carvana", "camvid"] 
    model_name = "segformer_ft_from_b2_cityscapes1024_rgb_img"
    mode = "predict" #["train", "predict"]

    if mode == "train":
        model, val_dl = trainModel(data_name, model_name)
        #valModel(data_name, val_dl, model)
    else:
        #load saved model from file
        model = f"./results/{data_name}/{model_name}/checkpoint-2400" 
        #test a pretrained model or pass a model for testing.
        testModel(data_name, model_name, model)


