
import config as conf
import data.diva_data  as data
import data.utils  as dut
import train as tm
import eval_model as em

""" Main file for DIVA data segmentation for virtual try-on """



def trainModel(dn):
    """ Train a segmentation model """

    cfg = conf.Config(dn, mode="train")
    train_dl, val_dl = data.loadData(cfg)
    dut.exploreData(train_dl)

    trainer = tm.TrainUNetSeg(cfg)
    model = trainer.trainModel(train_dl, val_dl)

    return model, val_dl


def testModel(dn, model=None):
    """ Test a model that is passed as input or a pretrained model """


    cfg = conf.Config(dn, mode="predict")
   
    #test_dl = data.getTestData(cfg)
    #em.evalDivaModel(cfg, test_dl, model, rle=True)

    _, val_dl = data.loadData(cfg)
    em.evalModel(cfg, val_dl, model)

    return


def valModel(dn, val, model=None):
    """ Validate a model with ground-truth """

    cfg = conf.Config(dn, mode="predict")
   
    em.evalModel(cfg, val_dl, model)

    return



if __name__ == "__main__":
   
   data_name = "diva"
   mode = "predict" #["train", "predict"]
   if mode == "train":
      model, val_dl = trainModel(data_name)
      valModel(data_name, val_dl, model)
   else:
     #model = None #load saved model from file
     #test a pretrained model or pass a model for testing.
     testModel(data_name)
   
  


