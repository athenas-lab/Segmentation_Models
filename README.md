## Image Segmentation Models

### Fully convolutional models
```unet```: Training and evaluation of following UNet models using different datasets.
- UNet: model as described in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  - UNet with batchnorm (can be optionally enabled in the config file by setting norm = "bn")
  
### Transformer-based models
```segformer```: Fine-tuning of Huggingface implementation of Segformer using different datasets.
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
Test split: 232 images, 32 classes

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Building|iou, acc Car|iou, acc LaneMkgsDriv|iou, acc Pedestrian|iou, acc Sidewalk|iou, acc SUVPickupTruck|iou, acc TrafficLight|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:-----------|:-------------------:|:-----------------:|:---------------:|:---------------------:|:-------------------:|
| Unet     |          |               |                  |
| Unet+ BN |          |               |                  |
| FPN      |          |               |                  |  
| Segformer| 0.419    |   0.495       | 0.892            | 0.867, 0.957     |0.792, 0.937|0.503, <br>0.584     |0.442, 0.618       |0.815, 0.938     |0.322, <br>0.506       |0.592, 0.736         | 
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### Binary-class

#### Carvana: Single car dataset
Validation split: binary classes (car, background)

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Car     | iou, acc background|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:------------------:|
| Unet     |          |               |                  |
| Unet+ BN |          |               |                  | 
| Segformer|   0.994  |  0.997        |    0.998         | 0.990, 0.996     | 0.997, 0.998       |


#### Diva: virtual try-on dataset
Test: 4774 samples, binary class (cloth, background)

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Cloth   | iou, acc background|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:------------------:|
| Unet     |          |               |                  |
| Unet+ BN |          |               |                  | 
| Segformer|   0.999  |  0.999        |    0.999         | 0.999, 0.999     | 0.999, 0.999       |


#### Salt: geological dataset
Test: 3990 samples, binary class (salt, background)

| Model    | mean IOU | mean accuracy | overall accuracy | iou, acc Salt   | iou, acc background|
|:---------|:--------:|:-------------:|:----------------:|:----------------:|:------------------:|
| Unet     |          |               |                  |
| Unet+ BN |          |               |                  | 
| Segformer|   0.975  |  0.987        |    0.991         | 0.963, 0.979     | 0.988, 0.994       |





### References and Acknowledgments
- [UNet implementation](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144)
- [Fusing Backbone Features with Feature Pyramid Network](https://medium.com/@freshtechyy/fusing-backbone-features-using-feature-pyramid-network-fpn-c652aa6a264b)
- [Segformer: Fine-Tune a Semantic Segmentation Model with a Custom Dataset](https://huggingface.co/blog/fine-tune-segformer)
- [Segformer Finetuning with CamVid dataset](https://github.com/CherifiImene/buildings_and_road_segmentation/blob/main/data_handler/data.py)

