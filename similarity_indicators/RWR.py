#coding=UTF-8
'''
Created on Nov 29, 2016

@author: ZWT
'''
import numpy as np
import time

import scipy


def RWR(MatrixAdjacency_Train):
    #similarity_StartTime = time.process_time()
    
    Parameter = 0.85

    S = sum(MatrixAdjacency_Train)
    S[S == 0] = 1e-10
    Matrix_TransitionProbobility = MatrixAdjacency_Train / S
    Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])
    
    Temp = Matrix_EYE - Parameter * Matrix_TransitionProbobility.T
    Temp = np.nan_to_num(Temp)
    INV_Temp = scipy.linalg.pinv(Temp)
    #INV_Temp = np.nan_to_num(INV_Temp)
    Matrix_RWR = (1 - Parameter) * np.dot(INV_Temp,Matrix_EYE)
    Matrix_similarity = Matrix_RWR + Matrix_RWR.T
    
    
    #similarity_EndTime = time.process_time()
    #print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity
