import numpy as np
import optuna

import Evaluation_Indicators
import similarity_indicators.NMF_LC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
# 计算auc方法
def CalcAUC(sim, test_pos, test_neg):
    # Extract similarity scores for positive and negative test edges
    print('sim',sim)
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()

    # Create labels for positive and negative scores
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # Combine positive and negative scores
    scores = np.concatenate([pos_scores, neg_scores])

    # Use roc_auc_score to calculate AUC
    auc = roc_auc_score(labels, scores)

    return auc

def objective(trial):
    global train_matrix
    global test_matrix
    global train_epochs
    global train_content
    global nt
    global test_pos
    global test_neg


    params = {
        'k': trial.suggest_int('k', 2, len(train_matrix)),
        'f': trial.suggest_float("f", 0, 2)

    }

    similarity_matrix = similarity_indicators.NMF_LP.NMF(train_matrix, train_content, params['k'],
                                                           train_epochs, params['f'], nt)

    # auc = Evaluation_Indicators.AUC.Calculation_AUC(train_matrix,
    #                                                 test_matrix,
    #                                                 similarity_matrix)

    auc = CalcAUC(similarity_matrix, test_pos, test_neg)

    return auc



def nmflc_optimizer(MatrixAdjacency_Train, MatrixAdjacency_Test, content, epochs, network_type,test_po, test_ne):
    global train_matrix, test_matrix, train_content, train_epochs, nt,test_pos,test_neg
    train_matrix = MatrixAdjacency_Train
    test_matrix = MatrixAdjacency_Test
    train_epochs = epochs
    train_content = content
    nt = network_type
    test_pos=test_po
    test_neg=test_ne

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))#sampler采样器方法
    study.optimize(objective, n_trials=100)#n_trials执行的试验次数

    trial = study.best_trial#最优超参对应的trial，有一些时间、超参、trial编号等信息；
    # print("Best trial Value: ", trial.value)

    # for key, value in trial.params.items():
    #     print("{}: {}".format(key, value))

    return trial.value, trial.params['k'], trial.params['f']


