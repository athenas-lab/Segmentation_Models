import numpy as np
import torchvision.transforms as T

""" Data utility functions """

def tensor2Img(x):
    """ Given a torch tensor, convert it to an image and display """

    #define the image transforms
    img_transf =  T.Compose([     
             T.Lambda(lambda t: t.permute(1,2,0)), #CHW --> HWC
             T.Lambda(lambda t: t*255.), #RGB
             T.Lambda(lambda t: t.numpy().astype(np.uint8)), #convert to img
             T.ToPILImage()])

    z = img_transf(x)         

    return z

def exploreData(dl):
    """ Preview the data """

    #batch = next(iter(dl))
    #print("Batch size = %d"%len(batch))
    for i, b in enumerate(dl):
        assert("mask" in b.keys())
        fn, img, msk = b["img_name"], b["image"], b["mask"]
        print(img.shape, msk.shape)
        x = tensor2Img(img[0])
        x.show()
        #mask = msk[0].unsqueeze(0)
        print(msk[0].shape)
        x = tensor2Img(msk[0])
        x.show()


        #if i == 1:
        return
