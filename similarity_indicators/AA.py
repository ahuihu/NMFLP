#coding=UTF-8
'''
Created on 2016��11��27��

@author: ZWT
'''
import numpy as np
import time

def AA(MatrixAdjacency_Train):
    #similarity_StartTime = time.process_time()
    S = sum(MatrixAdjacency_Train)
    S[S == 0] = 1e-10
    logTrain = np.log(S)
    logTrain = np.nan_to_num(logTrain)
    logTrain.shape = (logTrain.shape[0],1)
    logTrain[logTrain == 0] = 1e-10
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / logTrain
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)
    
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train_Log)

    #similarity_EndTime = time.process_time()
    #print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity