#import modules
from torchvision import datasets, transforms
import numpy as np
import scipy
from sklearn import metrics

val_transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_dataset =  datasets.ImageFolder('/Users/Simona/Fresno_Area/test', transform=val_transform)

test_emb_triplet = np.loadtxt('/Users/Simona/Fresno_Area/test_emb_triplet1')
print('MSE_COS_triplet')
MSE_COS(test_emb_triplet,test_dataset)
test_emb_raw = np.loadtxt('/Users/Simona/Fresno_Area/X_test_raw')
print('MSE_COS_raw')
MSE_COS(test_emb_raw,test_dataset)
test_emb_tile2vec = np.loadtxt('/Users/Simona/Fresno_Area/X_test_tile2vec')
print('MSE_COS_tile2vec')
MSE_COS(test_emb_tile2vec,test_dataset)
test_emb_resnet18 = np.loadtxt('/Users/Simona/Fresno_Area/X_test_resnet18imagenet')
print('MSE_COS_resnet18')
MSE_COS(test_emb_resnet18,test_dataset)
test_emb_resnet50 = np.loadtxt('/Users/Simona/Fresno_Area/X_test_resnet50resisc45')
print('MSE_COS_resnet50')
MSE_COS(test_emb_resnet50 ,test_dataset)
test_emb_pca = np.loadtxt('/Users/Simona/Fresno_Area/X_test_PCA')
print('MSE_COS_pca')
MSE_COS(test_emb_pca,test_dataset)
test_emb_kmeans = np.loadtxt('/Users/Simona/Fresno_Area/X_test_kmeans')
print('MSE_COS_kmeas')
MSE_COS(test_emb_kmeans,test_dataset)

def MSE_COS(test_emb,test_dataset):
    MSE_tot = []
    labels_tot = []
    COS_tot = []
    for i in range(0,len(test_emb),2):
        MSE_sameloc = []
        labels_list = []
        COS_sameloc = []
        for j in range(0,len(test_emb),2):
            #print(i,j+1)
            Feature_vec_drone_img = test_emb[i]
            Feature_vec_planet_img = test_emb[j+1]
            labels = tuple([test_dataset[i][1],test_dataset[j+1][1]])
            labels_list.append(labels)
            MSE_sameloc.append(metrics.mean_squared_error(Feature_vec_drone_img, Feature_vec_planet_img))
            COS_sameloc.append(1 - scipy.spatial.distance.cosine(Feature_vec_planet_img,Feature_vec_drone_img))
        MSE_tot.append(MSE_sameloc)
        labels_tot.append(labels_list)
        COS_tot.append(COS_sameloc)
    #MSE
    top_5_count_MSE = 0
    for i in range(0,len(MSE_tot)):
        list1 = np.asarray(MSE_tot[i])
        list2 = np.asarray(labels_tot[i])
        idx   = np.argsort(list1)
        list1 = np.array(list1)[idx]
        list2 = np.array(list2)[idx]
        top_5 = list1[0:5]
        top_5_l = list2[0:5]
        #print(top_5_l)
        for k in range(5):
            #print(k)
            if top_5_l[k][0] == top_5_l[k][1]:
                top_5_count_MSE += 1
    print("MSE accuracy",top_5_count_MSE/len(MSE_tot)*100)

    #COS
    top_5_count_COS = 0
    for i in range(len(COS_tot)):
        list1 = np.asarray(COS_tot[i])
        list2 = np.asarray(labels_tot[i])
        idx   = np.argsort(list1)
        list1 = np.array(list1)[idx]
        list2 = np.array(list2)[idx]
        list1 = np.flipud(list1)
        list2 = np.flipud(list2)
        top_5 = list1[0:5]
        top_5_l = list2[0:5]
        for k in range(5):
            if top_5_l[k][0] == top_5_l[k][1]:
                top_5_count_COS += 1
    print("COS accuracy",top_5_count_COS/len(COS_tot)*100)
