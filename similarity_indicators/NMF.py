import numpy as np
import time
from sklearn.decomposition import NMF
import networkx as nx
def nmf(MatrixAdjacency_Train):

    nmf_model = NMF(init="random", max_iter=100)#n_components保留样本特征  init随机 W H 的初始化方法
    U = nmf_model.fit_transform(MatrixAdjacency_Train)
    V = nmf_model.components_
    Matrix_similarity = np.dot(U,V)
    similarity_EndTime = time.process_time()
    # print("    SimilarityTime: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity