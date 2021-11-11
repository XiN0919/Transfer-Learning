# encoding=utf-8

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
for i in range(0, 4):
    for j in range(0, 4):
        if i != j:
            src, tar = '../data/' + domains[i], '../data/' + domains[j]
            src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
            # print(src_domain)
            Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs, Ys.ravel())
            y_pred = clf.predict(Xt)
            acc = sklearn.metrics.accuracy_score(Yt, y_pred)
            print(f'Accuracy : {acc:.3f}')
