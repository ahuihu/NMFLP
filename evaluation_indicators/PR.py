
import numpy as np
import time


'''先在Predict中去除训练集Train中1的数据（变为0），
   然后将Predict转为行向量并排序，将Test按Predict的排序方式排序。
   最后在排好序的Test中选择前L个数据，即为预测最大L个数据对应的真实值，
   其中1的个数为m，Precision = m / L'''
def Calculate_PR(Train, Test, Predict, L):
    NewPredict = (1 - Train) * Predict        #在Predict中去掉训练集中的1对应的预测值
    length = len(Train)
    NewPredict = NewPredict.reshape((length * length, 1))      #转为行向量
    Test = Test.reshape((length * length, 1))

    id = np.argsort(-NewPredict, axis = 0)
    #NewPredict = NewPredict[id]
    Test = Test[id]

    sum1 = 0
    for i in range(L):
        sum1 += Test[i][0][0]
    return sum1 / L

def Calculate_PR1(matrix_train,matrix_test,matrix_score,MaxNodeNum):
    '''
    Precision指标：m/L
    :param matrix_train:
    :param matrix_test:
    :param matrix_score:
    :param MaxNodeNum:
    :return:
    '''
    L = 200
    predict=[]
    for i in range(1,MaxNodeNum):
        for j in range(i,MaxNodeNum):
            if(i!=j and matrix_train[i][j]==0):
                # 如果节点i与j在训练集中无连边，就在预测集中追加该数据
                predict.append((i,j,matrix_score[i][j]))

    dtype = [('Node1', int), ('Node2', int), ('Score', float)]
    nm = np.array(predict, dtype=dtype)
    nm = np.sort(nm, order=['Score', ]) #按照Score对测试集排序

    # 选取预测集中评分最高的L条数据
    new_nm = nm[nm.shape[0] - L:nm.shape[0]]
    m=0
    for x in new_nm:
        # x = (node1,node2,score), 若预测的边（node1,node2)存在于测试集中，分数+1
        if matrix_test[x[0]][x[1]]>0:
            m=m+1
    precision = m/L
    print(precision)

    return precision