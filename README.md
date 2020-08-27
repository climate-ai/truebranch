# TrueBranch

TrueBranch is a novel way of distinguishing truthfully from untruthfully reported imagery by using low-resolution satellite data to verify the location of high-resolution drone data.

This project collects tif files from googel earth engine, transforms them to the right image format, trains a deep neural network on matching images with different resolutions coming from a different source and compares different classifiers.

In the next section different instructions are given for the different steps, following this workflow:

## Workflow:

1.	Getting tif files of specific area in google earth engine
2.	Tif to images locally (jupyter notebooks)
3.	Training and Testing Model
4.	Validation with Diagrams and Tensorboard
5.	MRS and COS Similarity Classifier 
6.	RF, MLP, LR Classifier

## Requiremens:
This project was built with a virtual environment, using conda. The requirements.txt lists all needed packages. 
The conda virtual environment file "truebranch.yml" can be downloaded and activated with the command: conda env create -f <environment-name>.yml (here: conda env create -f truebranch.yml)

## 1. Collecting Tif files from google earth engine - Folder "TIF_Files_GEE"
Naip tif images (representing high resolution drone images) and Sentinel tif images (at low resolution) are collected for the area xxx.
Script to collect Naip Tif files: NAIP_extraction.js 
Script to collect Sentinel Tif files: Sentinel_extraction.js
Extracted area defined inside scripts (here Central Valley, California, spanning latitudes [36:45;37:05] and longitudes [-120:25;-119:65].

## 2. Transforming Tif files to images, locally (jupyter notebooks)
Extracting 200x200 pixel png images out of a Tif file. This is done in a for loop where the tif file is cropped and the sentinel images upsampeled. The variable origin_path indicates the path where the tif files are saved, origin_path+"/"+temp_path defines where the png images are saved. The created dataset includes 2324 train images,
166 test images and 166 validation images.

## 3.	Training and Testing Model
The goal of this project is to train a model in order to be able to detect wheter two images from a different source with different resoltutions are from the same location or not. A CNN is trained and tested using the train, test and validation set created in the step above. The exact procedure is furthur explained in the Master Thesis "TrueBranch.pdf". The metric learning with Triplet loss is implemented using Kevin Musgrave's pytorch metric learning library (https://kevinmusgrave.github.io/pytorch-metric-learning/). If you use this code, please additionally cite his work:

@misc{musgrave2020pytorch,
    title={PyTorch Metric Learning},
    author={Kevin Musgrave and Serge Belongie and Ser-Nam Lim},
    year={2020},
    eprint={2008.09164},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

The metric learning code of this project can be found in the folder Metric learning, the script to create the training, validation and test embeddings is called "Metric_learning_tripletloss.py", the notebook including visualisations and tensorboard "Metric_Learning_Triplet_Loss.ipynb".
