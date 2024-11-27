## Image Segmentation Models

- UNet: model as described in the original UNet paper
- UNet with batchnorm (can be optionally enabled in the config file by setting norm = "bn"


 ## Datasets used for training and evaluation
 - Carvana: https://www.kaggle.com/competitions/carvana-image-masking-challenge
   - mask images with a single background and remove background
 - Salt: https://www.kaggle.com/c/tgs-salt-identification-challenge/overview
   - mask the areas with salt deposit
