# 2022-BackgroundMatting
This repository is for `Real-Time High-Resolution Background Matting, CVPR 2021`([pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)) reimplementation.  
I referred to the [official implementaion](https://github.com/PeterL1n/BackgroundMattingV2) in PyTorch.  
I used pretrained weights of DeepLabV3 from [VainF](https://github.com/VainF/DeepLabV3Plus-Pytorch).

## Requirements
I share anaconda environment yml file.
Create environment by `conda env create -n $ENV_NAME -f py38torch1110.yml`  
You can also check requirements from the yml file.


## Usage
### Training Base Network
The Base Network includes ASPP module from DeepLabV3. I used pretrained DeepLabV3 weight([best_deeplabv3_resnet50_voc_os16.pth](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0)).

```
usage: python train_base.py
```

This repo use Hydra for experiment configuration. The configuration file is under `./app/configs`.

For training base network, please set the corrensponding parameters in `./app/configs/train_base.yaml`.

```
arguments:
  checkpoint_path       checkpoint saving dir path
  logging_path          path to save logs
  batch_size            batch size
  num_workers           num workers
  epochs                epochs to train
  pretrained_model      pretrained model path

  defaults:
    data                configuration file which handling the dataset path configuration. See below.
```

For dataset path configuration, please refer to `./app/configs/data/default.yaml`

```
  original_work_dir     the root directory of the repository
  data_root             the root directory of the dataset

  rgb_data_dir          the directory of the rgb dataset
  bck_data_dir          the directory of the background dataset

  train_rgb_path        foreground data directory path for training
  train_alp_path        alpha matte data directory path for training
  valid_rgb_path        foreground data directory path for validation
  valid_alp_path        alpha matte data directory path for validation

  train_bck_path        background data directory path for training
  valid_bck_path        background data directory path for validation

```


### Training Whole Network (Refinement Network)
After training the Base Network, train the Base Network and Refinement Network jointly.

```
usage: python train_refine.py
```

For training refine network, please set the corrensponding parameters in `./app/configs/train_refine.yaml`

```
arguments:
  checkpoint_path       checkpoint saving dir path
  logging_path          path to save logs
  batch_size            batch size
  num_workers           num workers
  epochs                epochs to train
  pretrained_model      pretrained model path

  defaults:
    data                configuration file which handling \
                        the dataset path configuration.   \
                        Same as base training.
```

### Test Image Background Matting
You can download my trained weight form [here](https://drive.google.com/drive/folders/1UnoNk7fp44PyDsyfdnIc6-wAzNxP9xgn?usp=sharing).  
Using trained weight, you can test image background matting.  
Make sure that related image and background data are same order in each directory.

```
usage: python test_image.py
```

For tesing the network, please set the corrensponding parameters in `./app/configs/test_image.yaml`

```
  original_work_dir       the root directory of the repository
  pretrained_model        pretrained model path

  src_path                source directory path
  bck_path                background directory path
  output_path             output directory path
  output_type             choose output types from
                          [composite layer, alpha matte,\
                          foreground residual, error map,\
                          reference map]
```

## Datasets
Limited datasets are available on the [official website](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).

## Examples
|![5_src](https://user-images.githubusercontent.com/45582330/172120943-d560d03d-e7a7-4931-af07-b9e1f4da74ed.png)|![5_alp](https://user-images.githubusercontent.com/45582330/172120967-5283aee2-3654-48a1-b8d8-0414c7128e2c.jpg)|![5_com](https://user-images.githubusercontent.com/45582330/172121008-cc32b344-95bc-44d0-b649-be646bb54778.png)|
|---|---|---|
|![14_src](https://user-images.githubusercontent.com/45582330/172121415-de1a4ceb-5b23-44d1-b081-fd6a21520543.png)|![14_alp](https://user-images.githubusercontent.com/45582330/172121437-96f10d1d-828f-428c-9828-5e977d5d6ce4.jpg)|![14_com](https://user-images.githubusercontent.com/45582330/172121451-19b902eb-8ef7-403d-b44f-d17e557b3037.png)|
|source image|predicted alpha matte|predicted foreground|

## References
- S.Lin, A.Ryabtsev, S.Sengupta, B.Curless, S.Seitz, I.Kemelmacher-Shlizerman. "Real-Time High-Resolution Background Matting.", in CVPR, 2021. ([pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf))
- [Official Home Page](https://grail.cs.washington.edu/projects/background-matting-v2/#/)
- [Official implementation in PyTorch](https://github.com/PeterL1n/BackgroundMattingV2)
- [DeepLabV3 pretrained weights](https://github.com/VainF/DeepLabV3Plus-Pytorch)
- L.C.Chen, G.Papandreou, F.Schroff, H.Adam. "Rethinking Atrous Convolution for Semantic Image Segmentation.", in CVPR 2017. ([arxiv](https://arxiv.org/abs/1706.05587))
