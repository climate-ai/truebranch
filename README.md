# TrueBranch

TrueBranch is a novel way of distinguishing truthfully from untruthfully reported imagery by using low-resolution satellite data to verify the location of high-resolution drone data.

This project collects tif files from googel earth engine, transforms them to the right image format, trains a deep neural network on matching images with different resolutions coming from a different source and compares different classifiers.

In the next section different instructions are given for the different steps, following this workflow:

## Workflow:

1.	Getting tif files of specific area in google earth engine
2.	Tif to png transformation
3.	Training and Testing Model
4.	Visualisation of test embeddings with Diagrams
5.	MRS and COS Similarity Classifier 
6.	RF, MLP, LR Classifier

## Requiremens:
This project was built with a virtual environment, using conda. The requirements.txt lists all needed packages. 
The conda virtual environment file "truebranch.yml" can be downloaded and activated with the command: conda env create -f <environment-name>.yml (here: conda env create -f truebranch.yml)

## 1. Collecting Tif files from google earth engine - Folder "TIF_Files_GEE"
Naip tif images (representing high resolution drone images) and Sentinel tif images (at low resolution) are collected for the area xxx.
Script to collect Naip Tif files: NAIP_extraction.js 
Script to collect Sentinel Tif files: Sentinel_extraction.js
Extracted area defined inside scripts (here Central Valley, California, spanning latitudes [36:45;37:05] and longitudes [-120:25;-119:65]).
The tif files are exported to Google Drive. Here the extraced tif files can be found in the data folder.

## 2. Transforming Tif files to images
Extracting 200x200 pixel png images out of a Tif file. This is done in a for loop where the tif file is cropped and the sentinel images upsampeled. The variable origin_path indicates the path where the tif files are saved, origin_path+"/"+temp_path defines where the png images are saved. The created dataset includes 2324 train images,
166 test images and 166 validation images.
Here the images are stored in the data folder.

## 3.	Training and Testing Model
The goal of this project is to train a model in order to be able to detect wheter two images from a different source with different resoltutions are from the same location or not. A CNN is trained and tested using the train, test and validation set created in the step above. The exact procedure is furthur explained in the Master Thesis "TrueBranch.pdf". The metric learning with Triplet loss is implemented using Kevin Musgrave's pytorch metric learning library (https://kevinmusgrave.github.io/pytorch-metric-learning/). If you use this code, please additionally cite his work:

@misc{musgrave2020pytorch, \
    title={PyTorch Metric Learning}, \
    author={Kevin Musgrave and Serge Belongie and Ser-Nam Lim}, \
    year={2020}, \
    eprint={2008.09164}, \
    archivePrefix={arXiv}, \
    primaryClass={cs.CV} \
}

The metric learning code of this project can be found in the folder Metric learning, the script to create the training, validation and test embeddings is called "Metric_learning_tripletloss.py", the notebook including visualisations and tensorboard "Metric_Learning_Triplet_Loss.ipynb".

## 4. Visualisation of test embeddings with Diagrams
The resulted embeddings, representing the images in the trained feature space, are plotted using the seaborn package. The code for visualisation can be found in the folder Visualisation in the document "embed_visualisation.py".

The MSE distance plots for raw images, images in resnet45 feature space and in our feature space (as explained in chapter 4 of the Master Thesis) can be found in the notebook "Metric_Learning_Triplet_Loss.ipynb".

## 5.MRS and COS Similarity Classifier 
The Mean Square Error and COS Similarity are threshold based classifiers, evaluating the performance of different embeddings. The classifiers are applied to the embeddings created with Metric learning (as implemented above) as well as to a feature extraction baseline. The baseline is presented in the Thesis, the embeddings calculation for the baseline can be found in the script: applying_diff_models_to_data.py. The classifiers are implemented in "MSE_COS_accuracy.py" or as notebook "Top-5 accuracy.ipynb", both in the folder Classifier.

## 6.	RF, MLP, LR Classifier
The Random forest, Logistic Regression, Multi-layer Perceptron Classifiers are applied to the embeddings in order to evaluate the quality of the embeddings. The classifiers are implemented in the document "RF_LR_MLP_Classifier.py" inside the Classifier folder. Like above, beside the truebranch embeddings, the feature extraction baseline is implemented to compare these embeddings to others. The script "Applying_diff_Classifiers.py" calculates the accuracy for the different embeddings. 

All results obtained through this project can be found in chapter 5 of the Truebranch.pdf document. 
Resnet50_PCA_KMeans_RawImage_Classifiers
