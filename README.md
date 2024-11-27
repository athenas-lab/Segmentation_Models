## Image Segmentation Models

- UNet: model as described in the original UNet paper
- UNet with batchnorm (can be optionally enabled in the config file by setting norm = "bn"


## Datasets used for training and evaluation
- Carvana: https://www.kaggle.com/competitions/carvana-image-masking-challenge
   - mask images with a single car and remove background
- Salt: https://www.kaggle.com/c/tgs-salt-identification-challenge/overview
   - mask the areas with salt deposit

## Dependencies



## Training
- config.py has the data and training configuration for different datasets. You can extend this to 
add your own dataset.
- make sure to set the data_root and paths as per your confguration,
- The main_ files are the started files for different datasets. 
  - For e.g. to train on the carvana dataset: 
    - set mode = "train" in "main_carvana_seg.py"
    - change configuration if needed in "config.py"
    - run: "python main_carvana_seg.py" 

## Predicting masks
  - For e.g. to test on the carvana dataset: 
    - set mode = "predict" in "main_carvana_seg.py"
    - set the model path in "config.py"
    - run: "python main_carvana_seg.py" 





### References and Acknowledgments

1.  UNet implementation: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
2.  https://www.kaggle.com/competitions/carvana-image-masking-challenge

