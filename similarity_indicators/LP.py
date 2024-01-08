#coding=UTF-8
'''
Created on Nov 29, 2016

@author: ZWT
'''

import numpy as np
import time

def LP(Parameter, MatrixAdjacency_Train):
    similarity_StartTime = time.process_time()
    CN = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    
    # Parameter = 1
    L3 = np.dot(CN, MatrixAdjacency_Train)
    
    Matrix_similarity = CN + L3 * Parameter
    similarity_EndTime = time.process_time()
    #print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity
