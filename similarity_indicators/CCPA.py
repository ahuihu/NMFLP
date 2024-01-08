import networkx as nx
import numpy as np
import networkx as nx

def CCPA(Parameter, MatrixAdjacency_Train):
    Tr = MatrixAdjacency_Train.copy()
    G = nx.from_numpy_matrix(Tr)
    CN = np.dot(Tr, Tr)
    N = len(Tr)

    avg_length = list(nx.all_pairs_shortest_path_length(G))
    mat_length = []
    for i in range(N):
        row_length = avg_length[i][1]
        row_length_list = []
        for r in range(N):
            rl = row_length.setdefault(r, N)     #找不到索引默认值设为N
            rl = N if rl < 1 else rl
            row_length_list.append(rl)
        mat_length.append(row_length_list)

    Distance = np.mat(mat_length)   # 列表转矩阵


    Matrix_similarity = Parameter * CN + (1 - Parameter) * (N / Distance)

    # Matrix_similarity = (1 - Parameter) * (N / Distance)

    return Matrix_similarity

def CCPA1(MatrixAdjacency_Train):
    # 将矩阵转换为无向图
    G = nx.from_numpy_matrix(MatrixAdjacency_Train)

    # 定义共同邻居中心性的函数
    def common_neighbor_centrality_func(u, v):
        return len(list(nx.common_neighbors(G, u, v)))

    # 定义要计算共同邻居中心性的节点对列表
    ebunch = G.edges()

    # 计算共同邻居中心性，并将其转换为列表
    common_neighbor_centrality_gen = ((u, v, common_neighbor_centrality_func(u, v)) for u, v in ebunch)
    common_neighbor_centrality_list = list(common_neighbor_centrality_gen)

    # 获取图中所有节点的数量
    num_nodes = len(G.nodes())

    # 创建一个空的矩阵
    matrix = np.zeros((num_nodes, num_nodes))

    # 将共同邻居中心性结果填充到矩阵中
    for (u, v, centrality) in common_neighbor_centrality_list:
        matrix[u - 1, v - 1] = centrality
        matrix[v - 1, u - 1] = centrality

    return matrix

