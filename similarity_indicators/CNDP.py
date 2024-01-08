import networkx as nx
import numpy as np

def CNDP(beta, MatrixAdjacency_Train):
    Tr = MatrixAdjacency_Train.copy()
    G = nx.from_numpy_matrix(Tr)
    CN = np.dot(Tr, Tr)
    DegreeList = sum(Tr)
    N = len(Tr)
    C = nx.average_clustering(G)
    alpha = -(beta * C)

    Matrix_similarity = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1, N):
            if CN[i,j] == 0:            #没有共同邻居
                Matrix_similarity[i,j] = 0
                Matrix_similarity[j, i] = 0
            else:
                listNeighbor = list(nx.common_neighbors(G,i,j))
                sumDegree = 0
                for lN in listNeighbor:
                    sumDegree += DegreeList[lN] ** alpha
                Matrix_similarity[i, j] = (CN[i,j] + 2) * sumDegree
                Matrix_similarity[j, i] = Matrix_similarity[i, j]




    return Matrix_similarity