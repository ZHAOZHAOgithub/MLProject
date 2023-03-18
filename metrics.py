import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))




def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def Num_metric(pred, true):

    """数值预测常用指标"""
    mse = MSE(pred, true)
    mae = MAE(pred, true)

    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)


    return mse, mae, rmse, mape, mspe, rse



def Label_metric(precited,expected):

    """二分类混淆矩阵计算"""

    part = precited ^ expected  # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)  # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    tp_list = list(precited & expected)  # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)  # 将FP的计算结果转换为list
    tp = tp_list.count(1)  # 统计TP的个数
    fp = fp_list.count(1)  # 统计FP的个数
    tn = pcount[0] - tp  # 统计TN的个数
    fn = pcount[1] - fp  # 统计FN的个数
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    '''precited = list(precited)
    expected = list(expected)
    precision = precision_score(precited, expected, average='binary')
    recall = recall_score(precited, expected, average='binary')
    F1 = f1_score(precited, expected, average='binary')'''
    if tp != 0:
        precision = tp / (tp+fp)               # 精确率
        recall = tp / (tp+fn)                  # 召回率
        F1 = (2*precision*recall) / (precision+recall)    # F1
    else:
        precision = 0
        recall = 0
        F1 = 0
    return accuracy, precision, recall, F1


