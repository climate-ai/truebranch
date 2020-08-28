#import modules

from torchvision import datasets, transforms
import torch
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Set the image transforms
train_transform = transforms.Compose([transforms.Resize(50)])

val_transform = transforms.Compose([transforms.Resize(50)])

train_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/train', transform=train_transform)
val_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/val', transform=val_transform)
test_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/test', transform=val_transform)

# Resnet50 pretrained on Resisc45
# loading modelsclassifier_url = "https://tfhub.dev/google/remote_sensing/resisc45-resnet50/1"

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

# Embed tiles
n_tiles = len(train_dataset)
z_dim = 2048 #Model output embeddings size
X_train = np.zeros((n_tiles, z_dim))
for idx in range(0,n_tiles):
    tile = train_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile = classifier.predict(tile[np.newaxis, ...])
    X_train[idx,:] = tile

# Embed tiles
n_tiles = len(test_dataset)
z_dim = 2048  #Model output embeddings size
X_test = np.zeros((n_tiles, z_dim))
for idx in range(n_tiles):
    tile = test_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile = classifier.predict(tile[np.newaxis, ...])
    X_test[idx,:] = tile

np.savetxt('/Users/Simona/Fresno_Area/X_train_resnet50resisc45',X_train)
np.savetxt('/Users/Simona/Fresno_Area/X_test_resnet50resisc45',X_test)

#pca
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.9)

# Embed tiles
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

# Embed tiles
n_tiles = len(test_dataset)
z_dim = 3*50*50
X_test = np.zeros((n_tiles, z_dim))

for idx in range(n_tiles):
    tile = test_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile/255.0
    tile_2d = tile.reshape(z_dim)
    tile = tile_2d.reshape(-1, 1)
    tile = pca.fit_transform(tile)
    tile = np.moveaxis(tile, -1, 0)
    X_test[idx,:] = tile

np.savetxt('/Users/Simona/Fresno_Area/X_train_PCA',X_train)
np.savetxt('/Users/Simona/Fresno_Area/X_test_PCA',X_test)

#K_meansimport random
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy.linalg import norm
import random
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# data
data_size, num_clusters = len(train_dataset),int(len(train_dataset)/2)

# Embed tiles
n_tiles = len(train_dataset)
z_dim = 100
X_train = np.zeros((n_tiles,z_dim*z_dim*3))

for idx in range(n_tiles):
    tile = train_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile / 255.0
    tile = tile.reshape(z_dim*z_dim*3)
    tile = torch.from_numpy(tile)
    X_train[idx,:] = tile

# data
data_size, num_clusters = len(test_dataset),int(len(test_dataset)/2)

# Embed tiles
n_tiles = len(test_dataset)
z_dim = 100
X_test = np.zeros((n_tiles,z_dim*z_dim*3))
#print(X_train.shape)

for idx in range(n_tiles):
    tile = test_dataset[idx][0]
    tile = np.asarray(tile)
    tile = tile / 255.0
    #print(tile.shape)
    tile = tile.reshape(z_dim*z_dim*3)
    tile = torch.from_numpy(tile)
    X_test[idx,:] = tile

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=int(len(train_dataset)/2))
X_train_kmeans = kmeans.fit_transform(X_train)
np.savetxt('/content/drive/My Drive/X_train_kmeans',X_train_kmeans)

kmeans = KMeans(n_clusters=int(len(test_dataset)/2))
X_test_kmeans = kmeans.fit_transform(X_test)
np.savetxt('/content/drive/My Drive/X_test_kmeans',X_test_kmeans)

#Raw in_features
# Embed tiles
n_tiles = len(train_dataset)
z_dim = 3*50*50
X_train = np.zeros((n_tiles, z_dim))

for idx in range(n_tiles):
    tile = train_dataset[idx][0]
    tile = np.asarray(tile)
    tile_2d = tile.reshape(z_dim)
    tile = tile_2d.reshape(-1, 1)
    #print(tile)
    tile = np.moveaxis(tile, -1, 0)
    #print(tile.shape)
    X_train[idx,:] = tile

# Embed tiles
n_tiles = len(test_dataset)
z_dim = 3*50*50
X_test = np.zeros((n_tiles, z_dim))

for idx in range(n_tiles):
    tile = test_dataset[idx][0]
    tile = np.asarray(tile)
    print("1",tile.shape)
    tile_2d = tile.reshape(z_dim)
    print("2",tile_2d.shape)
    tile = tile_2d.reshape(-1, 1) #first dim stays the same, one dim added to second dim
    print("3",tile.shape)
    tile = np.moveaxis(tile, -1, 0)
    print("4",tile.shape)
    X_test[idx,:] = tile

np.savetxt('/Users/Simona/Fresno_Area/X_train_raw',X_train)
np.savetxt('/Users/Simona/Fresno_Area/X_test_raw',X_test)

#Tile2Vec

import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('/Users/Simona/tile2vec/src')
import tilenet
from tilenet import make_tilenet
import resnet
from resnet import ResNet18
#from src.tilenet import make_tilenet
#from src.resnet import ResNet18
import torchvision
from torchvision import datasets, transforms

#Loading pretrained models# Setting up model
in_channels = 4
z_dim = 512
cuda = torch.cuda.is_available()
# tilenet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# Use old model for now
tilenet = ResNet18()
if cuda: tilenet.cuda()

# Load parameters
model_fn = '../models/naip_trained.ckpt'
print(model_fn)
checkpoint = torch.load(model_fn,map_location=torch.device('cpu'))
tilenet.load_state_dict(checkpoint)
tilenet.eval()

#Embed NAIP tiles
from torchvision import datasets, transforms

# Set the image transforms
train_transform = transforms.Compose([transforms.Resize(100),
                                    #transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                                    #transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(100),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


train_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/train', transform=train_transform)
val_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/val', transform=val_transform)
test_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/test', transform=val_transform)

#Train Embeddings
# Embed tiles
from torch.autograd import Variable
n_tiles = len(train_dataset)
z_dim = 512
X_train = np.zeros((n_tiles, z_dim))
for idx in range(n_tiles):
    tile = train_dataset[idx][0]
    tile = tile.numpy()
    tile = np.concatenate((tile, np.zeros((1, 100, 100))), axis=0)
    tile = np.expand_dims(tile, axis=0)
    # Scale to [0, 1]
    tile = tile / 255
    # Embed tile
    tile = torch.from_numpy(tile).float()
    tile = Variable(tile)
    z = tilenet.encode(tile)
    z = z.data.numpy()
    #print(z.shape)
    X_train[idx,:] = z

np.savetxt('/Users/Simona/Fresno_Area/X_train_tile2vec',X_train)


# Embed tiles
from torch.autograd import Variable
n_tiles = len(test_dataset)
z_dim = 512
X_test = np.zeros((n_tiles, z_dim))
for idx in range(n_tiles):
    tile = test_dataset[idx][0]
    tile = tile.numpy()
    tile = np.concatenate((tile, np.zeros((1, 100, 100))), axis=0)
    tile = np.expand_dims(tile, axis=0)
    # Scale to [0, 1]
    tile = tile / 255
    # Embed tile
    tile = torch.from_numpy(tile).float()
    tile = Variable(tile)
    z = tilenet.encode(tile)
    z = z.data.numpy()
    #print(z.shape)
    X_test[idx,:] = z

np.savetxt('/Users/Simona/Fresno_Area/X_test_tile2vec',X_test)
