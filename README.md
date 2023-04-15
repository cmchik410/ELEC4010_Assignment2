# ELEC4010_Assignment2

This repositories contains codes for the two tasks in ELEC4010 Assignemt 2.

The first task is to implement a ResNet-50 for classifying the state of lesion (i.e, benign / malignant)

The second task is to implement a Unet for brain MRI segmentation.

## Dataset
The dataset used in task 1 could be downloaded at the followings:

Training image:
https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip

Training Annotation:
https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv

Testing image:
https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip

Testing Annotation:
https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv


The dataset used in task 2 could be downloaded at 
https://drive.google.com/file/d/1166Qg5U807OpQGhZ2oz4l6C2jAsoU3Ts/view

$ tree
datasets
├── ISBI2016
│   ├── testing
|   |   ├──...
│   └── training
|   |   ├──...
|   └── test_label.csv
|   └── train_label.csv
|   MRI
└── └── ...

$ ./tree-md .
# Project tree

.
 * [tree-md](./tree-md)
 * [dir2](./dir2)
   * [file21.ext](./dir2/file21.ext)
   * [file22.ext](./dir2/file22.ext)
   * [file23.ext](./dir2/file23.ext)
 * [dir1](./dir1)
   * [file11.ext](./dir1/file11.ext)
   * [file12.ext](./dir1/file12.ext)
 * [file_in_root.ext](./file_in_root.ext)
 * [README.md](./README.md)
 * [dir3](./dir3)


## Requirement
The following packages are used for the tasks.

```bash
python==3.9.16
pytorch==2.0
torchvision=0.15.0
medpy==0.4.0
tqdm
yaml
```

## How to use

Run following in terminal for training Q1 model

```bash
python Q1.py
```

Run following in terminal for training Q2 model

```bash
python Q2.py
```

 
