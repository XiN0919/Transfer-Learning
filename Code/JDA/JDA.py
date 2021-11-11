"""
    说明：本程序是JDA文献的代码复现
    参考：王晋东github: https://github.com/jindongwang/transferlearning/blob/master/code/traditional/JDA/JDA.py
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))  # n_feas*ns水平拼接n_feas*nt ————》 n_feas*(ns+nt)
        X /= np.linalg.norm(X, axis=0)  # 按列处理二范数,列/范数 归一化
        m, n = X.shape  # n_feas*(ns+nt)  m-->fea_number n-->data size
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1/ns*np.ones((ns, 1)), -1/nt*np.ones((nt, 1))))   # 行方向拼接  2081*1
        C = len(np.unique(Ys))  # 去重复，并排序输出：即标签的类目
        H = np.eye(n)-1/n*np.ones((n, n))   # 2081*2081

        M = 0
        Y_tar_pseudo = None
        # training epoch
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C  # 2081*2081
            print(M0.shape)
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            print(M)
            M = M / np.linalg.norm(M, 'fro')  # 矩阵元素绝对值平方和开根号
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            # KNN模型
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc


if __name__ == '__main__':
    # four source domains or target domains
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    # create source domains and target domains using for
    for i in range(1):
        for j in range(2):
            if i != j:
                src, tar = '../data/'+domains[i], '../data/'+domains[j]
                # load mat information
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                jda = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
                acc, y_pred, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
