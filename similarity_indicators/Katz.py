#coding=UTF-8
'''
Created on Nov 29, 2016

@author: ZWT
'''
import numpy as np
import time

from scipy.linalg import svd


# def Katz(Parameter, MatrixAdjacency_Train):
#     similarity_StartTime = time.process_time()
#
#     Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])
#     Temp = Matrix_EYE - Parameter * MatrixAdjacency_Train
#
#     u, s, vh = svd(Temp, full_matrices=False)
#
#     # 设置一个阈值，将小于该阈值的奇异值设为零，以避免除零错误
#     threshold = 1e-10
#     s_inv = np.where(s > threshold, 1.0 / s, 0.0)
#
#     Matrix_similarity = vh.T @ np.diag(s_inv) @ u.T
#
#     Matrix_similarity = Matrix_similarity - Matrix_EYE
#
#     similarity_EndTime = time.process_time()
#     # print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
#
#     return Matrix_similarity



def Katz(Parameter, MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()
    Parameter = 0.01
    Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])
    Temp = Matrix_EYE - MatrixAdjacency_Train * Parameter

    Matrix_similarity = np.linalg.pinv(Temp)

    Matrix_similarity = Matrix_similarity - Matrix_EYE

    similarity_EndTime = time.process_time()
    similarity_StartTime = time.process_time()
    # Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])
    # Temp = Matrix_EYE - Parameter * MatrixAdjacency_Train
    #
    # Matrix_similarity = np.linalg.pinv(Temp)
    #
    # Matrix_similarity = Matrix_similarity - Matrix_EYE
    #
    # similarity_EndTime = time.process_time()
    # # print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    # return Matrix_similarity
    #print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity



