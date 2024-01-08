#coding=UTF-8
'''
Created on 2016��11��19��

@author: ZWT
'''
import numpy as np
import time

def Calculation_AUC(MatrixAdjacency_Train,MatrixAdjacency_Test,Matrix_similarity):

    # print('    Calculation AUC......')
    AUCnum = 100000
    np.random.seed(0)
    NodeNum = MatrixAdjacency_Train.shape[0]
    # 生成Matrix_similarity的上三角矩阵。此处貌似做复杂了，更简单的做法应该为：Matrix_similarity = np.triu(Matrix_similarity)
    Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * MatrixAdjacency_Train)

    # Matrix_NoExist 矩阵中将本来没有边的元素设置为1，其余为0
    Matrix_NoExist = np.ones(NodeNum) - MatrixAdjacency_Train - MatrixAdjacency_Test - np.eye(NodeNum)
    # 得到测试集和不存在边的矩阵的上三角矩阵
    Test = np.triu(MatrixAdjacency_Test)
    NoExist = np.triu(Matrix_NoExist)

    # 分别得到测试集中边的个数以及不存在的边的个数
    Test_num = len(np.argwhere(Test == 1))
    NoExist_num = len(np.argwhere(NoExist == 1))
    # print('Test_num：%d'%Test_num)
    # print('NoExist_num：%d'%NoExist_num)


    # 随机生成测试集边与不存在边的下标
    Test_rd = [int(x) for index,x in enumerate((Test_num * np.random.rand(1,AUCnum))[0])]
    NoExist_rd = [int(x) for index,x in enumerate((NoExist_num * np.random.rand(1,AUCnum))[0])]

    # TestPre与NoExistPre分别为仅含有测试集边的分数值与不存在边分数值的矩阵
    TestPre= Matrix_similarity * Test
    NoExistPre = Matrix_similarity * NoExist

    # Test_Data与NoExist_Data分别为存有测试集边的分数值与不存在边分数值的一维数组
    TestIndex = np.argwhere(Test == 1)
    Test_Data = np.array([TestPre[x[0],x[1]] for index,x in enumerate(TestIndex)]).T
    NoExistIndex = np.argwhere(NoExist == 1)
    NoExist_Data = np.array([NoExistPre[x[0],x[1]] for index,x in enumerate(NoExistIndex)]).T

    # Test_rd和NoExist_rd为Test_Data与NoExist_Data数组中随机下标所对应的分数值
    Test_rd = np.array([Test_Data[x] for index,x in enumerate(Test_rd)])
    NoExist_rd = np.array([NoExist_Data[x] for index,x in enumerate(NoExist_rd)])

    n1,n2 = 0,0
    for num in range(AUCnum):
        if Test_rd[num] > NoExist_rd[num]:
            n1 += 1
        elif Test_rd[num] == NoExist_rd[num]:
            n2 += 0.5
        else:
            n1 += 0
    auc = float(n1+n2)/AUCnum

    return auc