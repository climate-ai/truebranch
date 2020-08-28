#import modules

import RF_LR_MLP_Classifier
from RF_LR_MLP_Classifier import RF
from RF_LR_MLP_Classifier import LR
from RF_LR_MLP_Classifier import MLP
from RF_LR_MLP_Classifier import SVC

from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from PIL import Image
import logging
import faiss
import matplotlib.pyplot as plt
from cycler import cycler
import record_keeper
import pytorch_metric_learning
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)

from record_keeper import RecordKeeper, RecordWriter

from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

from sklearn.decomposition import PCA

def RF_LR_MLP(train_emb,test_emb):
    #scaling data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(train_emb) #Fit to data, then transform it.
    X_test_minmax = min_max_scaler.transform(test_emb)
    print('RF',RF(X_train_minmax, X_test_minmax))
    print('LR',LR(X_train_minmax, X_test_minmax))
    print('MLP',MLP(X_train_minmax, X_test_minmax))

#Loading TrueBranch Embeddings
#Load embeddings - 64-dimensional
train_emb = np.loadtxt('/Users/Simona/Fresno_Area/train_emb_triplet')
test_emb = np.loadtxt('/Users/Simona/Fresno_Area/test_emb_triplet')
print("Classifier with Truebranch embed - 64dim")
RF_LR_MLP(train_emb,test_emb)
#Load embeddings - 1000-dimensional
train_emb = np.loadtxt('/Users/Simona/Fresno_Area/train_emb_triplet1')
test_emb = np.loadtxt('/Users/Simona/Fresno_Area/test_emb_triplet1')
print("Classifier with Truebranch embed - 1000dim")
RF_LR_MLP(train_emb,test_emb)

#Tile2Vec embeddings
train_emb  = np.loadtxt('/Users/Simona/Fresno_Area/X_train_tile2vec')
test_emb  = np.loadtxt('/Users/Simona/Fresno_Area/X_test_tile2vec')
print("Classifier with Tile2Vec embed")
RF_LR_MLP(train_emb,test_emb)

# Resnet50 pretrained on Resisc45
# load embed
train_emb= np.loadtxt('/Users/Simona/Fresno_Area/X_train_resnet50resisc45')
test_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_test_resnet50resisc45')
print("Classifier with Resnet50 embed")
RF_LR_MLP(train_emb,test_emb)

# Resnet18 pretrained on Imagenet
train_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_train_resnet18imagenet')
test_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_test_resnet18imagenet')
print("Classifier with Resnet18 embed")
RF_LR_MLP(train_emb,test_emb)

#PCA
#different image size
train_transform = transforms.Compose([transforms.Resize(50)])
val_transform = transforms.Compose([transforms.Resize(50)])
train_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/train', transform=train_transform)
val_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/val', transform=val_transform)
test_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/test', transform=val_transform)
pca = PCA(n_components = 0.9)

# Creating X_train
n_tiles = len(train_dataset)
z_dim = 3*50*50
X_train = np.zeros((n_tiles, z_dim))

for idx in range(n_tiles):
    tile = train_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile_2d = tile.reshape(z_dim)
    tile = tile_2d.reshape(-1, 1)
    tile = pca.fit_transform(tile)
    tile = np.moveaxis(tile, -1, 0)
    X_train[idx,:] = tile

# Creating X_test
n_tiles = len(test_dataset)
z_dim = 3*50*50
X_test = np.zeros((n_tiles, z_dim))

for idx in range(n_tiles):
    tile = test_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile_2d = tile.reshape(z_dim)
    tile = tile_2d.reshape(-1, 1)
    tile = pca.transform(tile)
    #tile = pca.fit_transform(tile)
    tile = np.moveaxis(tile, -1, 0)
    X_test[idx,:] = tile

np.savetxt('/Users/Simona/Fresno_Area/X_train_PCA',X_train)
np.savetxt('/Users/Simona/Fresno_Area/X_test_PCA',X_test)

train_emb= np.loadtxt('/Users/Simona/Fresno_Area/X_train_PCA')
test_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_test_PCA')
print("Classifier with PCA embed")
RF_LR_MLP(train_emb,test_emb)

#K_means
train_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_train_kmeans')
test_emb  = np.loadtxt('/Users/Simona/Fresno_Area/X_test_kmeans')
print("Classifier with k-means embed")
RF_LR_MLP(train_emb,test_emb)

#Raw in_features
# Embed tiles
#Transformations without "to tensor"!
n_tiles = len(train_dataset)
z_dim = 3*50*50
X_train = np.zeros((n_tiles, z_dim))

for idx in range(len(train_dataset)):
    tile = train_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile_2d = tile.reshape(z_dim)
    tile = tile_2d.reshape(-1, 1)
    tile = np.moveaxis(tile, -1, 0)
    X_train[idx,:] = tile

n_tiles = len(test_dataset)
z_dim = 3*50*50
X_test = np.zeros((n_tiles, z_dim))

for idx in range(len(test_dataset)):
    tile = test_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile_2d = tile.reshape(z_dim)
    tile = tile_2d.reshape(-1, 1)
    tile = np.moveaxis(tile, -1, 0)
    X_test[idx,:] = tile

np.savetxt('/Users/Simona/Fresno_Area/X_train_rawfloat',X_train)
np.savetxt('/Users/Simona/Fresno_Area/X_test_rawfloat',X_test)

train_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_test_rawfloat')
test_emb = np.loadtxt('/Users/Simona/Fresno_Area/X_train_rawfloat')
print("Classifier with raw image embed")
RF_LR_MLP(train_emb,test_emb)
