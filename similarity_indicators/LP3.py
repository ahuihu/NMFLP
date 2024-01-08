#coding=UTF-8
'''
Created on Nov 29, 2016

@author: ZWT
'''

import numpy as np
import time

def LP3(MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()
    Mid = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)
    Matrix_similarity = np.dot(MatrixAdjacency_Train, Mid)
    similarity_EndTime = time.process_time()
    #print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity
