import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn.decomposition import PCA


def Normalization(A):        #对矩阵A进行归一化
    row = len(A)
    col = len(A[0])
    max_A = A.max()
    min_A = A.min()
    maxA = np.ones((row, col)) * max_A
    minA = np.ones((row, col)) * min_A
    A = (A - minA) / (maxA - minA)
    return A


def NMF(MatrixAdjacency_Train, attribute_matrix, nk, N, f, network_type):     #  Sxy = beta * A * (E - beta^2 * A^2).I
    # C = generateNodeFeatureByA(MatrixAdjacency_Train)
    C= attribute_matrix
    Matrix_similarity = training(MatrixAdjacency_Train, C, nk, N, f, network_type)
    return Matrix_similarity

def training(A, C1, nk, epochs, f, network_type):
    if network_type == 1:
        norm = 'l1'
        C1 = Normalization(C1)
        pca = PCA(n_components=nk)
        C = pca.fit_transform(C1)
    if network_type == 0:
        norm = 'l2'
        C = C1
    n=len(A)

    dim_pca = C.shape[1]
    se = 1
    np.random.seed(se)
    U = np.random.rand(n, nk)       #生成U,V,Y的初始随机值
    np.random.seed(se)
    V = np.random.rand(nk,n)
    np.random.seed(se)
    Y =np.random.rand(dim_pca,nk)
    Loss=[]
    for i in range(epochs):     #将U,V和Y交替迭代N次直到收敛
        #更新U
        f1 = np.dot(A, V.T) + f * np.dot(C, Y)
        f2 = np.dot(np.dot(U, V), V.T) + f * U
        # f2[f2==0] = 1e-10
        # delta = f1/ f2
        delta = (f1 / (f2 + 1e-10)) + 1e-10
        U = U * delta
        U = preprocessing.normalize(U, norm=norm)


        # 更新V
        f1 = np.dot(U.T, A)
        f2 = np.dot(np.dot(U.T, U), V)
        # f2[f2==0] = 1e-105210
        # delta = f1 / f2
        delta = (f1 / (f2 + 1e-10)) + 1e-10
        V = V * delta
        V = preprocessing.normalize(V, norm=norm)

        # 更新Y
        f1 = np.dot(C.T, U)
        f2 = np.dot(np.dot(C.T, C), Y)
        # f2[f2==0] = 1e-10
        # delta = f1 / f2
        delta = (f1 / (f2 + 1e-10)) + 1e-10
        Y = Y * delta
        Y = preprocessing.normalize(Y, norm=norm)
        # print("Loss:{}".format(calculate_cost(A, U, V, Y, C, f)))

    # print(U.shape)
    # simP = np.dot(np.dot(C, Y), V)
    # print(simP)
    simP = np.dot(U, V)
    return simP


def calculate_cost(A, U, V, Y, C, f):
    # A = self._create_base_matrix(self.graph)
    loss_1 = np.linalg.norm(A - U.dot(V), ord="fro") ** 2
    loss_2 = np.linalg.norm(U - C.dot(Y), ord="fro") ** 2
    loss_all = loss_1 + f*loss_2
    # print("{}======{}======{}".format(loss_1, loss_2, loss_all))
    return loss_all