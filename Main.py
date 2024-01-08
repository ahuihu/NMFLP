# coding=UTF-8
'''
Created on 2016??11??20??

@author: ZWT
'''

import time

import networkx as nx
import numpy as np
import pandas as pd
from dgl.data import PPIDataset
from dgl.data import TexasDataset
from dgl.data import WisconsinDataset
from torch.linalg import LinAlgError

import Evaluation_Indicators.AUC
import Evaluation_Indicators.PR
import Initialize2022
import similarity_indicators.ACT
import similarity_indicators.Jaccard
import similarity_indicators.AA
import similarity_indicators.RA
import similarity_indicators.NMF
import similarity_indicators.CCPA
import similarity_indicators.SVD
import similarity_indicators.CommonNeighbor
import similarity_indicators.Katz
import similarity_indicators.LP3
import similarity_indicators.RWR
from NetworkLoader import NetworkLoader
from similarity_indicators.NMF_LC_Optimizer import nmflc_optimizer

startTime = time.process_time()

def Normalization(A):        #对矩阵A进行归一化
    row = len(A)
    col = len(A[0])
    max_A = A.max()
    min_A = A.min()
    maxA = np.ones((row, col)) * max_A
    minA = np.ones((row, col)) * min_A
    A = (A - minA) / (maxA - minA)
    return A
def generateNodeFeatureByA(MatrixAdjacency_Train):
    Tr = MatrixAdjacency_Train.copy()
    X1 = Tr
    X2 = np.dot(MatrixAdjacency_Train, Tr)
    X3 = np.dot(X2, Tr)

    X_list = [X1, X2, X3]

    i = 1
    for Xi in X_list:
        Xi = Normalization(Xi)

        if i == 1:
            X_max = Xi
        else:
            X_max = np.hstack((Xi, X_max))  # numpy.hstack(tup),其中tup是数组序列，表示在水平方向上平铺
        i += 1
    return X_max

def cov(X, Y):#协方差
    n = np.shape(X)[0]  # 特诊个数
    X, Y = np.array(X), np.array(Y)
    meanX, meanY = np.mean(X), np.mean(Y)
    cov = sum(np.multiply(X - meanX, Y - meanY)) / (n - 1)
    return cov
def ccovmat(content_matrix):
    simMatrix = content_matrix.copy()# 样本集
    covmat1=[]
    na=np.shape(simMatrix)[0]# 特征attr总数
    n=np.shape(simMatrix)[1]
    covmat1 = np.full((na, n), fill_value=0.)
    for i in range(na):
        for j in range(n):
            covmat1[i, j] = cov(simMatrix[:, i], simMatrix[:, j])
    return covmat1


pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行




normal_networks = ['dt']
attribute_networks = ['cornell'] 

networks = normal_networks + attribute_networks

from dgl.data import CornellDataset
print(networks)

cls = ['CN', 'AA', 'RA', 'Jaccard', 'LP3', 'Katz', 'CCPA', 'RWR', 'ACT', 'NMF', 'SVD', 'NMFLC_A', 'NMFLC_T', 'Katz_beta', 'NK1', 'F1', 'EPOCH1', 'NK2', 'F2', 'EPOCH2']
cls1 = ['network'] + cls

count = 0

auc_finalResults = pd.DataFrame(columns=cls1)
finalResults = pd.DataFrame(columns=cls1)
pre_finalResults = pd.DataFrame(columns=cls1)

def evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L):
    auc = Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train,
                                                    MatrixAdjacency_Test,
                                                    Matrix_similarity)
    pre = Evaluation_Indicators.PR.Calculate_PR(MatrixAdjacency_Train, MatrixAdjacency_Test,
                                                    Matrix_similarity, L)
    return auc, pre


for network in networks:

    print("Network: {}".format(network))

    aucResults = pd.DataFrame(columns=cls)
    preResults = pd.DataFrame(columns=cls)

    netFile = r'Data\\' + network + '.txt'
    # print("******************{}****************".format(network))
    if network in ['leadership', 'revolution', 'crime', 'opsahl-ucforum', 'membership', 'dt'] or network.find('bipartite_') != -1:
        Graph, left_num, right_num = Initialize2022.load_bipartiteNetwork(netFile)
    elif network in ['cornell', 'cora', 'cornell', 'citeseer', 'polblog', 'terrorist', 'texas', 'washington', 'wisconsin']  or network.find('attribute_') != -1:
        loader = NetworkLoader(network)
        adjajency_matrix, content_matrix, node_community_label_list, edge_list = loader.network_parser('Data/input/' + network)
        Graph = nx.from_numpy_array(adjajency_matrix)
        content = content_matrix
    elif network in ['TexasDataset','WisconsinDataset', "PPIDataset", "CornellDataset"]:
        if network == 'TexasDataset':
            dataset = TexasDataset()
        elif network == 'WisconsinDataset':
            dataset = WisconsinDataset()
        elif network == "PPIDataset":
            dataset = PPIDataset(mode='valid')
        elif network == "CornellDataset":
            dataset = CornellDataset()
        g = dataset[0]
        Graph = g.to_networkx().to_undirected()
        feat = g.ndata["feat"]
        # 将 PyTorch 张量转换为 NumPy 数组
        feat_matrix_np = feat.numpy()
        content = np.asarray(feat_matrix_np)
    else:
        Graph = Initialize2022.load_Network(netFile)

    L = len(Graph.edges)

    MatrixAdjacency_TrainSets, MatrixAdjacency_TestSets = Initialize2022.train_test_split(Graph, 10)

    for fold in range(len(MatrixAdjacency_TrainSets)):

        MatrixAdjacency_Train = MatrixAdjacency_TrainSets[fold]
        print(MatrixAdjacency_Train.shape)
        MatrixAdjacency_Test = MatrixAdjacency_TestSets[fold]

        # print('----------CN----------')
        Matrix_similarity = similarity_indicators.CommonNeighbor.Cn(MatrixAdjacency_Train)
        print(MatrixAdjacency_Train.shape)
        print(Matrix_similarity.shape)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'CN'] = auc
        preResults.loc[fold, 'CN'] = pre

        # print('----------ACT----------')
        Matrix_similarity = similarity_indicators.ACT.ACT(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'ACT'] = auc
        preResults.loc[fold, 'ACT'] = pre

        # print('----------Jaccard----------')
        Matrix_similarity = similarity_indicators.Jaccard.Jaccard(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'Jaccard'] = auc
        preResults.loc[fold, 'Jaccard'] = pre

        # print('----------AA----------')
        Matrix_similarity = similarity_indicators.AA.AA(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'AA'] = auc
        preResults.loc[fold, 'AA'] = pre

        # print('----------RA----------')
        Matrix_similarity = similarity_indicators.RA.RA(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'RA'] = auc
        preResults.loc[fold, 'RA'] = pre

        # print('----------CCPA----------')
        Matrix_similarity = similarity_indicators.CCPA.CCPA1(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'CCPA'] = auc
        preResults.loc[fold, 'CCPA'] = pre

        # print('----------NMF----------')
        Matrix_similarity = similarity_indicators.NMF.nmf(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'NMF'] = auc
        preResults.loc[fold, 'NMF'] = pre

        # print('----------SVD----------')
        Matrix_similarity = similarity_indicators.SVD.SVD(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'SVD'] = auc
        preResults.loc[fold, 'SVD'] = pre

        # print('----------LP3----------')
        Matrix_similarity = similarity_indicators.LP3.LP3(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'LP3'] = auc
        preResults.loc[fold, 'LP3'] = pre

        # print('----------Katz----------')
        katzAUCMax, pre, katz_beta = 0, 0, 0
        for b in np.linspace(0, 1, 101):
            try:
                # print('----------Katz----------')
                Matrix_similarity = similarity_indicators.Katz.Katz(b, MatrixAdjacency_Train)
                katzAuc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
            except LinAlgError:
                print("Katz有异常！")
                continue

            if katzAuc > katzAUCMax:
                katzAUCMax = katzAuc
                katz_beta = b
        aucResults.loc[fold, 'Katz'] = katzAUCMax
        aucResults.loc[fold, 'Katz_beta'] = katz_beta
        preResults.loc[fold, 'Katz'] = pre

        # print('----------RWR----------')
        Matrix_similarity = similarity_indicators.RWR.RWR(MatrixAdjacency_Train)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
        aucResults.loc[fold, 'RWR'] = auc
        preResults.loc[fold, 'RWR'] = pre

        # print('----------NMFLC_T----------')
        epochs = 100
        content1 = MatrixAdjacency_Train
        content2 = ccovmat(content1)
        auc, k, f = nmflc_optimizer(MatrixAdjacency_Train, MatrixAdjacency_Test, content2, epochs, 0)
        Matrix_similarity = similarity_indicators.NMF_LP.NMF(MatrixAdjacency_Train, content2, k,
                                                             epochs, f, 0)
        auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)

        aucResults.loc[fold, 'NMFLC_T'] = auc
        preResults.loc[fold, 'NMFLC_T'] = pre
        aucResults.loc[fold, 'F1'] = f
        aucResults.loc[fold, 'NK1'] = k
        aucResults.loc[fold, 'EPOCH1'] = epochs

        # print('----------NMFLC_A----------')
        if network in attribute_networks:
            auc, k, f = nmflc_optimizer(MatrixAdjacency_Train, MatrixAdjacency_Test, content, epochs, 1)
            Matrix_similarity = similarity_indicators.NMF_LP.NMF(MatrixAdjacency_Train, content, k,
                                                                 epochs, f, 1)
            auc, pre = evaluate(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, L)
            aucResults.loc[fold, 'NMFLC_A'] = auc
            aucResults.loc[fold, 'F2'] = f
            aucResults.loc[fold, 'NK2'] = k
            aucResults.loc[fold, 'EPOCH2'] = epochs
            preResults.loc[fold, 'NMFLC_A'] = pre

        print("===================={} Fold {} AUC Results====================".format(network, fold))
        print(aucResults.loc[fold])
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        aucResults.to_csv('results-2023\\auc_'  +network+ time_str + '.csv')
        print("===================={} Fold {} Precision Results====================".format(network, fold))
        print(preResults.loc[fold])


    print("====================={} AUC results=====================".format(network))
    df1 = aucResults.mean()
    aucResults.loc[10] = aucResults.mean()
    print(aucResults)
    df1.loc['network'] = network


    auc_finalResults.loc[count] = df1

    aucResults.to_csv('results-2023\\auc\\auc_' + network + '.csv')

    print("====================={} Precision results=====================".format(network))
    df2 = preResults.mean()
    preResults.loc[10] = preResults.mean()
    print(preResults)
    df2.loc['network'] = network

    pre_finalResults.loc[count] = df2


    # preResults.to_csv('results-2023\\precision\\pre_' + network + '.csv')
    count = count + 1


auc_finalResults.to_csv('results-2023\\auc_finalResults.csv')
# pre_finalResults.to_csv('results-2023\\pre_finalResults.csv')
