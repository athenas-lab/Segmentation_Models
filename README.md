## Image Segmentation Models

This repo contains the pytorch implementation of the following models. These models have been trained and evaluated on the datasets listed below for binary and multi-class segmentation. These models are briefly described on [this wiki](https://github.com/athenas-lab/Segmentation_Models/wiki).

### Fully convolutional models
- ```fully_conv_net```: Training and evaluation of the following fully convolutional networks using different segmentation datasets.
  - UNet: model as described in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  - UNet with batchnorm (can be optionally enabled in the config file by setting norm = "bn")
  - Feature Pyramid Network: [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144)
  
### Transformer-based models
- ```segformer```: Fine-tuning of Huggingface implementation of Segformer using different datasets.
 - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## Datasets used for training and evaluation

- ### Multi-class segmentation
* [Cambridge Video Dataset](https://www.kaggle.com/datasets/carlolepelaars/camvid): urban driving dataset
  - RGB images for urban streets, RGB masks for 32 classes
 
- ### Binary class segmentation
- [Carvana](https://www.kaggle.com/competitions/carvana-image-masking-challenge)
   - a dataset with single car images in RGB,  black/white masks (0: background, 1: car)
   - mask the car area and remove background
 
- DIVA: virtual-tryon
  - RGB images, grayscale mask to detect the areas with clothes/dress (0: background,  255: cloth)
     
- [Salt](https://www.kaggle.com/c/tgs-salt-identification-challenge/overview): geological dataset to detect salt
   - grayscale image, grayscale mask to detect the areas with salt deposit (0: no salt, 255: salt)


## Dependencies
```
- pip install torch torchvision
- pip install mathplotlib pandas
- pip install torcheval
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
- Test split: 232 images, 32 classes.
- Model trained using cross-entropy loss.

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Building|iou, acc Car|iou, acc LaneMkgsDriv|iou, acc Pedestrian|iou, acc Sidewalk|iou, acc SUVPickupTruck|iou, acc TrafficLight|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:-----------|:-------------------:|:-----------------:|:---------------:|:---------------------:|:-------------------:|
| Unet     | 0.093    |   0.127       | 0.660            | 0.49, 0.913      |0.019, 0.020|0, 0             |0, 0               |0.321, 0.505     |0,  0             |0, 0                 | 
| Unet+ BN | 0.144    |   0.191       | 0.748            | 0.648, 0.888     |0.388, 0.815|0, 0             |0, 0               |0.593, 0.771     |0.0, 0.0          |0.0, 0.0             |
| FPN      | 0.275    |   0.351       | 0.799            | 0.711, 0.838     |0.613, 0.894|0.284, 0.352     |0.168, 0.281       |0.697, 0.870     |0.064, 0.067       |0.282, 0.326         |    
| Segformer| 0.419    |   0.495       | 0.892            | 0.867, 0.957     |0.792, 0.937|0.503, 0.584     |0.442, 0.618       |0.815, 0.938     |0.322, 0.506       |0.592, 0.736         | 

- Segformer outperforms the fully convolutional models across all metrics for this dataset.
- Among the fully convolutional models, FPN outperforms the UNet model.
- Using batchnorm with UNet yields better performance.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Binary-class

#### Carvana: Single car dataset
Validation split: binary classes (car, background)

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Car     | iou, acc background|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:------------------:|
| Unet     |   0.97   | 0.985         |    0.99          | 0.953, 0.977     | 0.987, 0.993       |
| Unet+ BN |   0.975  | 0.988         |    0.992         | 0.96,  0.978     | 0.989, 0.994       |
| FPN      |   0.989  | 0.995         |    0.996         | 0.983, 0.993     | 0.995, 0.997       | 
| Segformer|   0.994  | 0.997         |    0.998         | 0.990, 0.996     | 0.997, 0.998       |

- Segformer outperforms the fully convolutional models across all metrics for this dataset.
- Among the fully convolutional models, FPN outperforms the UNet model.
- Unet with and without batchnorm provide comparable performance.
- FPN and Unet peform better with Cross-entropy loss compared to BCELogitsLoss.
- Unet with BN peforms slightly better with BCELogitsLoss loss compared to Cross-entropy loss.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### Diva: virtual try-on dataset
- Test: 4774 samples, binary class (cloth, background)
- Model trained using categorical cross-entropy loss (with 2 classes) yielded better performance than binary cross-entropy loss (with single class). 

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Cloth   | iou, acc background|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:------------------:|
| Unet     |   0.884  |  0.938        |    0.947         | 0.842, 0.913     | 0.926, 0.962       |
| Unet+ BN |   0.872  |  0.933        |    0.941         | 0.827, 0.912     | 0.917, 0.954       |
| FPN      |   0.992  |  0.996        |    0.997         | 0.99,  0.995     | 0.995, 0.997       |
| Segformer|   0.999  |  0.999        |    0.999         | 0.999, 0.999     | 0.999, 0.999       |

- Segformer outperforms the fully convolutional models across all metrics for this dataset.
- Among the fully convolutional models, FPN outperforms the UNet model.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### Salt: geological dataset
Test: 3990 samples, binary class (salt, background)

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Salt   | iou, acc background|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:------------------:|
| Unet(ce) |   0.579  |  0.719        |    0.759         | 0.697, 0.859     | 0.46, 0.578        |
| Unet+ BN |   0.64   |  0.771        |    0.799         | 0.737, 0.871     | 0.544, 0.671       |
| FPN      |   0.764  |  0.861        |    0.877         | 0.828, 0.917     | 0.7, 0.806         |  
| Segformer|   0.975  |  0.987        |    0.991         | 0.963, 0.979     | 0.988, 0.994       |

- Segformer outperforms the fully convolutional models across all metrics for this dataset.
- Among the fully convolutional models, FPN outperforms the UNet model.
- Using batchnorm with UNet yields better performance.
- Unet peforms better with Cross-entropy loss compared to BCELogitsLoss.
- FPN and Unet with BN peforms better with BCELogitsLoss loss compared to Cross-entropy loss.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### References and Acknowledgments
- [UNet implementation](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)

- [Fusing Backbone Features with Feature Pyramid Network](https://medium.com/@freshtechyy/fusing-backbone-features-using-feature-pyramid-network-fpn-c652aa6a264b)
- [Segformer: Fine-Tune a Semantic Segmentation Model with a Custom Dataset](https://huggingface.co/blog/fine-tune-segformer)
- [Segformer Finetuning with CamVid dataset](https://github.com/CherifiImene/buildings_and_road_segmentation/blob/main/data_handler/data.py)

