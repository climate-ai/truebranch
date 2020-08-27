import random
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy.linalg import norm
import random
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


def build_pairs_training(train_emb):

    train_pairs = train_emb.shape[0]
    same = []
    same_labels = []

    for i in range(0,train_pairs,2):
        #print(i,i+1)
        same.append(np.concatenate((train_emb[i], train_emb[i+1])))
        same_labels.append(1)
    #print("same",len(same))

    #naip even num, sen odd num
    diff = []
    diff_labels = []

    for i in range(0,train_pairs,2):
        j = random.randrange(1,(train_pairs-2),2)
        if i != j and abs(i-j)>1:
            #print(i,j)
            diff.append(np.concatenate((train_emb[i], train_emb[j])))
            diff_labels.append(0)
        else:
            j +=2
            #print("diff",i,j)
            diff.append(np.concatenate((train_emb[i], train_emb[j])))
            diff_labels.append(0)

    train_emb= same+diff
    train_lab = same_labels+diff_labels

    train_emb  = np.asarray(train_emb)
    train_lab   = np.asarray(train_lab)
    indices = np.arange(train_emb.shape[0])
    np.random.shuffle(indices)

    train_emb  = train_emb[indices]
    train_lab  = train_lab[indices]

    return train_emb , train_lab

def build_pairs_testing(test_emb):

    same = []
    same_labels = []

    test_pairs = test_emb.shape[0]

    for i in range(0,test_pairs,2):
        same.append(np.concatenate((test_emb[i], test_emb[i+1])))
        same_labels.append(1)

    #naip even num, sen odd num
    diff = []
    diff_labels = []

    for i in range(0,test_pairs,2):
        j = random.randrange(1,(test_pairs-2),2)
        if i != j and abs(i-j)>1:
            #print(i,j)
            diff.append(np.concatenate((test_emb[i], test_emb[j])))
            diff_labels.append(0)
        else:
            j +=2
            #print("diff",i,j)
            diff.append(np.concatenate((test_emb[i], test_emb[j])))
            diff_labels.append(0)

    test_emb= same+diff
    test_lab = same_labels+diff_labels

    test_emb = np.asarray(test_emb)
    test_lab  = np.asarray(test_lab)
    indices = np.arange(test_emb.shape[0])
    np.random.shuffle(indices)

    test_emb = test_emb[indices]
    test_lab = test_lab[indices]

    #test_emb = test_emb.reshape(-1, 1)
    #test_emb=  test_emb.reshape(-1, 1)

    return test_emb, test_lab

def RF(X_train,X_test,n_trials = 100):
    accs = np.zeros((n_trials,))
    for i in range(n_trials):
        # Splitting data and training RF classifer
        X_tr, y_tr = build_pairs_training(X_train)
        X_te, y_te =  build_pairs_testing(X_test)
        rf = RandomForestClassifier()
        rf.fit(X_tr, y_tr)
        accs[i] = rf.score(X_te, y_te)
    print('Mean accuracy: {:0.4f}'.format(accs.mean()))
    print('Standard deviation: {:0.4f}'.format(accs.std()))

def SVC(X_train,X_test,n_trials = 100):
    accs = np.zeros((n_trials,))
    for i in range(n_trials):
        # Splitting data and training RF classifer
        X_tr, y_tr = build_pairs_training(X_train)
        X_te, y_te =  build_pairs_testing(X_test)
        #reg = LogisticRegression(max_iter = 1000)
        reg = LinearSVC(max_iter=2000)
        reg.fit(X_tr, y_tr)
        accs[i] = reg.score(X_te, y_te)
    print('Mean accuracy: {:0.4f}'.format(accs.mean()))
    print('Standard deviation: {:0.4f}'.format(accs.std()))


def LR(X_train,X_test,n_trials = 100):
    accs = np.zeros((n_trials,))
    for i in range(n_trials):
        # Splitting data and training RF classifer
        X_tr, y_tr = build_pairs_training(X_train)
        X_te, y_te =  build_pairs_testing(X_test)
        #reg = LogisticRegression(max_iter = 1000)
        reg = SGDClassifier(loss='log',max_iter=1000)
        reg.fit(X_tr, y_tr)
        accs[i] = reg.score(X_te, y_te)
    print('Mean accuracy: {:0.4f}'.format(accs.mean()))
    print('Standard deviation: {:0.4f}'.format(accs.std()))

def MLP(X_train,X_test,n_trials = 100):
    accs = np.zeros((n_trials,))
    for i in range(n_trials):
        # Splitting data and training RF classifer
        X_tr, y_tr = build_pairs_training(X_train)
        X_te, y_te =  build_pairs_testing(X_test)
        clf = MLPClassifier(max_iter=1000)
        clf.fit(X_tr, y_tr)
        accs[i] = clf.score(X_te, y_te)
    print('Mean accuracy: {:0.4f}'.format(accs.mean()))
    print('Standard deviation: {:0.4f}'.format(accs.std()))
