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
