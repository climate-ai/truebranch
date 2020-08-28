#imports
import seaborn as sns, numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from pylab import *
import sys

#loading the learned embeddings
train_emb = np.loadtxt('../embeddings/train_emb_triplet')
train_lab = np.loadtxt('../embeddings/train_lab_triplet')
val_emb = np.loadtxt('../embeddings/val_emb_triplet')
val_lab = np.loadtxt('../embeddings/val_lab_triplet')
test_emb = np.loadtxt('../embeddings/test_emb_triplet')
test_lab = np.loadtxt('../embeddings/test_lab_triplet')

#Distance Plots
distance_to_ = []
for i in range(0,test_emb.shape[0]):
  sublist = []
  for j in range(0,test_emb.shape[0]):
    labels = tuple([int(test_lab[i]),int(test_lab[j])])
    sublist.append(tuple([norm(test_emb[i]-test_emb[j]),labels]))
    #sublist.sort()
  distance_to_.append(sublist)

 # List of distances between Sentinel (anchor) and Naip (pos,neg)
 # Plotting anchor, positive and negative distances

x_neg = []
x_pos = []
x_arr_neg = np.asarray([0]*len(test_dataset))
x_arr_pos = np.asarray([0]*len(test_dataset))
x_arr_neg_list = []
x_arr_pos_list = []
x_arr_labels_list = []
x_arr_labels_list_n = []

for i in range(0,int(len(distance_to_)/2)):
  x_n_ = []
  x_p_ = []
  x_p_labels = []
  x_n_labels = []
  x_n_labels_raw = []
  for j in range(0,int(len(distance_to_)/2)):
    if distance_to_[i*2+1][j*2][1][1] == i:
      x_p_.append(distance_to_[i*2+1][j*2][0])
      x_p_labels.append(i)
    else:
      x_n_.append(distance_to_[i*2+1][j*2][0])
      x_n_labels.append((i*2+1,j*2))
      x_n_labels_raw.append((i,j))
  x_arr_neg_ = np.asarray(x_n_)
  x_arr_pos_ = np.asarray(x_p_)
  x_arr_neg_list.append(x_arr_neg_)
  #x_arr_pos = x_arr_pos + x_arr_pos_
  x_arr_pos_list.append(x_arr_pos_)
  x_arr_labels_list.append(x_p_labels)
  x_arr_labels_list_n.append(x_n_labels_raw)
  x_mean_neg = np.mean(x_arr_neg, axis=0)
  x_mean_pos = np.mean(x_arr_pos, axis=0)
  x_std_neg = np.std(x_arr_neg, axis=0)
  x_std_pos = np.std(x_arr_pos, axis=0)

sns.set()
x = x_arr_neg_list
x_2 =  x_arr_pos_list
ax = sns.distplot(x,label="different location")
ax = sns.distplot(x_2,label="same location")
ax.set(xlabel='Distance', ylabel='Distribution')
plt.title("Distance between locations")
plt.legend()
plt.show()

#Distance of all Embeddings

x = [] #distances from Sentinel to all Naip embeddings as list
x_same = [] #distances Sentinel to Naip for same location
for i in range(0,int(len(distance_to_)/2)):
  x_ = []
  x_same_=[]
  for j in range(0,int(len(distance_to_)/2)):
    x_.append(distance_to_[i*2+1][j*2][0]) #distance from Sentinel to naip
    if distance_to_[i*2+1][j*2][1][1] == i:
      x_same_.append(distance_to_[i*2+1][j*2][0])
  x.append(x_)
  x_same.append(x_same_)

distances_to_plot = [0,1,2,3,4,5,6,7,8]
classes = int(len(distance_to_)/2)
y = list(range(0, int(len(distance_to_)/2)))

fig = plt.figure(figsize=(30,15))
subplots_adjust(hspace=1.000)
number_of_subplots=len(distances_to_plot)

fig, axs = plt.subplots(len(distances_to_plot), 1, constrained_layout=True,sharex=True)
axs = axs.ravel()
for i in range(0,len(distances_to_plot)):
  y = [distances_to_plot[i]]*classes
  axs[i].scatter(x[distances_to_plot[i]],y,label="negative images")
  #axs[i].scatter(x[distances_to_plot[i]],y, label='Distance between images of different locations')
  #axs[i].set_xlabel('Distance between images of different locations')
  axs[i].set_ylabel(str(distances_to_plot[i]))
  axs[i].plot(x_same[distances_to_plot[i]][0],distances_to_plot[i],'ro',markersize=10,label="positive images")
  axs[i].set_yticklabels([])
  #legend(loc="upper right")
#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='right')
fig.suptitle('Distances between anchor (Sentinel images) to positive/negative (NAIP images)', fontsize=10)
plt.xlabel('Distance between images of different locations')
#fig.labels()
#plt.ylabel('Location of Anchor image')
plt.show()
