import numpy as np


def select_K_by_variance_explained(sigma, threshold=0.9):#根据方差解释率选择K值
    """
    根据方差解释率选择K值
    :param sigma: 奇异值数组
    :param threshold: 方差解释率阈值
    :return: 最佳的K值
    """
    total_variance = np.sum(sigma ** 2)
    variance_explained = np.cumsum(sigma ** 2) / total_variance
    return np.argmax(variance_explained >= threshold) + 1

def SVD(MatrixAdjacency_Train):
    """
    完整奇异值分解矩阵分解方法
    :param R: 原始的邻接矩阵（二维数组），其中0表示缺失的连接
    :return: 预测的邻接矩阵
    """
    # 进行奇异值分解
    U, sigma, VT = np.linalg.svd(MatrixAdjacency_Train)

    # 选择K值降维
    K = select_K_by_variance_explained(sigma, threshold=0.9)#threshold=0.9是方差解释率的阈值。方差解释率是前K个奇异值所解释的总方差占全部奇异值解释的总方差的比例。我们将选择K值使得前K个奇异值所解释的总方差占所有奇异值解释的总方差的90%以上。

    # 使用选择的K值进行矩阵分解
    U_K = U[:, :K]
    sigma_K = np.diag(sigma[:K])
    VT_K = VT[:K, :]
    Matrix_similarity = np.dot(np.dot(U_K, sigma_K), VT_K)

    return Matrix_similarity