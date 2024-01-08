import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# 该方法用以将一个二分网络的数据文件加载为Graph（networkx）
def load_bipartiteNetwork(file_path):
    data = np.loadtxt(file_path)
    df = pd.DataFrame(data[:, 0:2], columns=['left', 'right'], dtype=int)

    #  对二分网络左部节点重新进行编号，其编号范围为：0~左部节点数目-1。比如左部有100个节点，其节点编号为0-99。
    left_id = df[['left']].drop_duplicates()  # 用户id
    left_id['left_id'] = np.arange(len(left_id))  # 创建新列new_uid，为uid编号[0,len(user_id)]
    df = df.merge(left_id, on=['left'])

    # 对二分网络右部节点重新进行编号，使其从左部节点数开始编起。比如左部有100个节点，其节点编号为0-99，则右部节点从100号开始编起。
    right_id = df[['right']].drop_duplicates()
    right_id['right_id'] = np.arange(len(left_id), len(left_id) + len(right_id))  # 创建新列new_uid，为uid编号[0,len(user_id)]
    df = df.merge(right_id, on=['right'])

    # 从dataf中选出新编的节点对编号
    node_pairs = df[['left_id', 'right_id']].values
    node_pair_list = node_pairs.tolist()

    totalNodeNum = len(left_id) + len(right_id)

    # 按照边的编号得到邻接矩阵
    matrixAdjacency = np.zeros([totalNodeNum, totalNodeNum])
    for item in node_pair_list:
        start = item[0]
        end = item[1]
        matrixAdjacency[start, end] = 1
        matrixAdjacency[end, start] = 1

    # 生成Graph
    Graph = nx.from_numpy_array(matrixAdjacency)
    return Graph, len(left_id), len(right_id)


# 该方法用以将一个以边为记录的网络数据文件加载为Graph。注意，数据中节点编号需要从1开始！
def load_Network(file_path):
    data = np.loadtxt(file_path)
    df = pd.DataFrame(data[:, 0:2], columns=['left', 'right'], dtype=int)
    # 从dataf中选出新编的节点对编号
    node_pairs = df.values-1
    totalNodeNum = np.max(df.values)
    node_pair_list = node_pairs.tolist()
    # 按照边的编号得到邻接矩阵
    matrixAdjacency = np.zeros([totalNodeNum, totalNodeNum])
    for item in node_pair_list:
        start = item[0]
        end = item[1]
        matrixAdjacency[start, end] = 1
        matrixAdjacency[end, start] = 1

    # 生成Graph
    Graph = nx.from_numpy_array(matrixAdjacency)
    return Graph

def edges_to_features_labels(edges, non_edges):
    # 将存在边和不存在边的列表合并为一个列表
    all_edges = edges + non_edges

    # 创建对应的标签，存在边标记为1，不存在边标记为0
    labels = [1] * len(edges) + [0] * len(non_edges)

    # 创建特征矩阵，这里简单地将边的起始节点作为特征
    features = [(edge[0], edge[1]) for edge in all_edges]

    return features, labels
# 将网络按照K折交叉验证方法划分成n_splits个训练集和测试集。
# 其返回值分别为n_splits个训练集和测试集的数组
def train_test_split(Graph, n_splits):
    totalNodeNum = nx.number_of_nodes(Graph)
    # 得到网络的边的列表
    edge_list = list(Graph.edges(data=False))


    # 按照边将数据集进行K折划分
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100)
    np.random.seed(1)
    y = np.random.randint(0, 2, len(edge_list))

    matrixAdjacency_TrainSets = []
    matrixAdjacency_TestSets = []

    train_edges_sets = []  # 存储每个训练集的存在边的列表
    test_edges_sets = []  # 存储每个测试集的存在边的列表
    train_non_edges_sets = []  # 存储每个训练集的不存在边的列表
    test_non_edges_sets = []  # 存储每个测试集的不存在边的列表

    for trainIndex, testIndex in kfold.split(edge_list, y):
        matrixAdjacency_Train = np.zeros([totalNodeNum, totalNodeNum])
        matrixAdjacency_Test = np.zeros([totalNodeNum, totalNodeNum])
        train_edges = []  # 当前训练集的存在边列表
        test_edges = []  # 当前测试集的存在边列表
        for i in trainIndex:
            item = edge_list[i]
            start = item[0]
            end = item[1]
            matrixAdjacency_Train[start, end] = 1
            matrixAdjacency_Train[end, start] = 1
            train_edges.append((start, end))
        matrixAdjacency_TrainSets.append(matrixAdjacency_Train)

        for i in testIndex:
            item = edge_list[i]
            start = item[0]
            end = item[1]
            matrixAdjacency_Test[start, end] = 1
            matrixAdjacency_Test[end, start] = 1
            test_edges.append((start, end))

        # 通过差集找到不存在边的列表
        all_edges = set(edge_list)
        train_non_edges = list(all_edges - set(train_edges))
        test_non_edges = list(all_edges - set(test_edges))
        matrixAdjacency_TestSets.append(matrixAdjacency_Test)

        train_edges_sets.append(train_edges)
        test_edges_sets.append(test_edges)
        train_non_edges_sets.append(train_non_edges)
        test_non_edges_sets.append(test_non_edges)
        # 将每个训练集和测试集的存在边和不存在边转换为特征矩阵和标签

    X_train, y_train = edges_to_features_labels(train_edges_sets, train_non_edges_sets)
    X_test, y_test = edges_to_features_labels(test_edges_sets, test_non_edges_sets)

    # return matrixAdjacency_TrainSets, matrixAdjacency_TestSets, X_train, X_test, y_train, y_test

    return matrixAdjacency_TrainSets, matrixAdjacency_TestSets, train_edges_sets, test_edges_sets, train_non_edges_sets, test_non_edges_sets
    # return matrixAdjacency_TrainSets, matrixAdjacency_TestSets

def get_test_pos_neg(matrixAdjacency_TestSets, test_edges_sets, test_non_edges_sets):
    test_pos_list = []  # list to store positive test sets for each fold
    test_neg_list = []  # list to store negative test sets for each fold

    for i in range(len(matrixAdjacency_TestSets)):
        matrixAdjacency_Test = matrixAdjacency_TestSets[i]
        test_edges = test_edges_sets[i]
        test_non_edges = test_non_edges_sets[i]

        # Extract positive test set (existing edges)
        test_pos = np.where(matrixAdjacency_Test == 1)
        test_pos_edges = list(zip(test_pos[0], test_pos[1]))

        # Extract negative test set (non-existing edges)
        test_neg_edges = test_non_edges

        test_pos_list.append(test_pos_edges)
        test_neg_list.append(test_neg_edges)

    return test_pos_list, test_neg_list
