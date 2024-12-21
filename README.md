## Image Segmentation Models

#### Fully convolutional models
```unet```: Training and evaluation of following UNet models using different datasets.
- UNet: model as described in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  - UNet with batchnorm (can be optionally enabled in the config file by setting norm = "bn")
  
#### Transformer-based models
```segformer```: Fine-tuning of Huggingface implementation of Segformer using different datasets.
- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## Datasets used for training and evaluation
- [Carvana](https://www.kaggle.com/competitions/carvana-image-masking-challenge)
   - mask images with a single car and remove background
- [Salt](https://www.kaggle.com/c/tgs-salt-identification-challenge/overview)
   - mask the areas with salt deposit

## Dependencies
```
- pip install torch torchvision
- pip install mathplotlib pandas
- pip isntall torcheval
```



## Training
- config.py has the data and training configuration for different datasets. You can extend this to 
add your own dataset.
- make sure to set the data_root and paths as per your confguration,
- The main_ files are the started files for different datasets. 
  - For e.g. to train on the carvana dataset: 
    ```
    - set mode = "train" in "main_carvana_seg.py"
    - change configuration if needed in "config.py"
    - run: "python main_carvana_seg.py"
    ```  

## Predicting masks
  - For e.g. to test on the carvana dataset:
    ```
    - set mode = "predict" in "main_carvana_seg.py"
    - set the model path in "config.py"
    - run: "python main_carvana_seg.py"
    ```

## Performance Evaluation

### Multi-class

#### Cambridge video data: Driving dataset. 
Test split: 232 images

| Model    | mean IOU | mean accuracy | overall accuracy | 
|:---------|:--------:|:-------------:|:----------------:|
| Unet     |          |               |                  |
| Unet+ BN |          |               |                  |
| FPN      |          |               |                  |  
| Segformer| 0.419    |   0.495       | 0.892            |   
----------------------------------------------------------


### Binary-class

#### Carvana: Driving dataset

| Model    | mean IOU | mean accuracy | Foreground class IOU |
|:---------|:--------:|:-------------:|:----------------------|
| Unet     |          |               |                       |
| Unet+ BN |          |               |                       | 
| Segformer|          |               |                       |


#### Diva: virtual try-on dataset

| Model    | mean IOU | mean accuracy | 
|:---------|:--------:|:-------------:|
| Unet     |          |               |
| Unet+ BN |          |               |  
| Segformer|          |               | 


#### Salt: geological dataset

| Model    | mean IOU | mean accuracy | 
|:---------|:--------:|:-------------:|
| Unet     |          |               |
| Unet+ BN |          |               |  
| Segformer|          |               | 






### References and Acknowledgments
- [UNet implementation](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144)
- [Fusing Backbone Features with Feature Pyramid Network](https://medium.com/@freshtechyy/fusing-backbone-features-using-feature-pyramid-network-fpn-c652aa6a264b)
- [Segformer: Fine-Tune a Semantic Segmentation Model with a Custom Dataset](https://huggingface.co/blog/fine-tune-segformer)
- [Segformer Fintuning with CamVid dataset](https://github.com/CherifiImene/buildings_and_road_segmentation/blob/main/data_handler/data.py)

