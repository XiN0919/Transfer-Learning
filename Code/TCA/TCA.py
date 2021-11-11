# encoding=utf-8
"""
    TCA论文复现
    借鉴：王晋东Github：https://github.com/jindongwang/transferlearning/blob/master/code/traditional/TCA/TCA.py
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=0.1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma


    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        # n_feas*ns水平拼接n_feas*nt ————》 n_feas*(ns+nt)
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)  # 按列处理二范数,列/范数 归一化
        m, n = X.shape  # n_feas*(ns+nt)  m-->fea_number n-->data size
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))  # 行方向拼接  （ns+nt）*1
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')  # M=M/M的F范数
        H = np.eye(n) - 1 / n * np.ones((n, n))  # 中心矩阵
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)  # 核函数 primal 相当于没改变X
        n_eye = m if self.kernel_type == 'primal' else n
        # a = KMK_T相当于原文 KLK_T; b=KHK_T
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        # a = np.linalg.inv(a)
        w, V = scipy.linalg.eig(a, b)  # 求特征值w和右特征向量
        ind = np.argsort(w)  # 特征值排序 返回对应索引
        A = V[:, ind[:self.dim]] # 前dim维的特征向量
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                src, tar = '../data/' + domains[i], '../data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                # print(src_domain)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                print("--------------------------")

                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt, Yt)
                print("--------------------------")
                tca2 = TCA(kernel_type='rbf', dim=30, lamb=1, gamma=1)
                acc2, ypre2 = tca2.fit_predict(Xs, Ys, Xt, Yt)
                print("--------------------------")

                print(f'Accuracy : {acc1:.3f}')
                print(f'Accuracy : {acc2:.3f}')