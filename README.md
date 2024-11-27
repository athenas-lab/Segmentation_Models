## Image Segmentation Models

- UNet: model as described in the original UNet paper
- UNet with batchnorm (can be optionally enabled in the config file by setting norm = "bn"


 ## Datasets used for training and evaluation
 - Carvana: https://www.kaggle.com/competitions/carvana-image-masking-challenge
   - mask images with a single car and remove background
 - Salt: https://www.kaggle.com/c/tgs-salt-identification-challenge/overview
   - mask the areas with salt deposit



### References and Acknowledgments

1.  UNet implementation: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
2.  https://www.kaggle.com/competitions/carvana-image-masking-challenge

