
import sys
sys.path.append("../")
import config as conf
import data.salt_data  as data
import data.utils  as dut
import train as tm
import eval_model as em

""" 
Main file for SALT data segmentation to identify areas of salt deposit.
https://www.kaggle.com/c/tgs-salt-identification-challenge/overview

"""


def trainModel(dn):
    """ Train a segmentation model """

    cfg = conf.Config(dn, mode="train")
    train_dl, val_dl = data.loadData(cfg)
    #dut.exploreData(train_dl)

    trainer = tm.TrainUNetSeg(cfg)
    model = trainer.trainModel(train_dl, val_dl)

    return model, val_dl

def testModel(dn, model=None):
    """ Test a model that is passed as input or a pretrained model """

    cfg = conf.Config(dn, mode="predict")

    #img_paths, mask_paths = data.getTestData(cfg)
    #em.evalModel(cf, img_paths, mask_paths, model=None)

    test_dl = data.getTestData(cfg)
    print(len(test_dl))
    em.evalModel(cfg, test_dl, model, rle=False)


    return


def valModel(dn, val, model=None):
    """ Validate a model with ground-truth """

    cfg = conf.Config(dn, mode="predict")
   
    em.evalModel(cfg, val_dl, model)

    return





if __name__ == "__main__":

   data_name = "salt"
   mode = "train" #["train", "predict"]

   if mode == "train":
      model, val_dl = trainModel(data_name)
      #eval an input model with GT masks
      valModel(data_name, model)
   else:   
      #test a pretrained model or pass a model for testing.
      testModel(data_name)

