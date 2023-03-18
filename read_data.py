from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



class Data(Dataset):
    """封装训练阶段数据集"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]
        self.shape = [x.shape,y.shape]

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len



class Data2(Dataset):
    """封装插补阶段数据集"""
    def __init__(self, x):
        self.x = x
        self.len = x.shape[0]
        self.shape = x.shape

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len



def load(path):
    """加载数据"""
    train_paths = '../data/'+path+'/training.xlsx'
    impute_paths = '../data/'+path+'/test.xlsx'
    try:
        train_data = pd.read_excel(train_paths,header=None) # 默认读取第一个sheet
        impute_data = pd.read_excel(impute_paths,header=None)
    except:
        print('Error in data path!!!')
    train = train_data.values[1:,:]
    #test = train_data.values[1:,-1]
    impute = impute_data.values[1:,:-1]
    train = np.array(train,dtype=np.float32)
    impute = np.array(impute, dtype=np.float32)
    for i in range(0,train.shape[1]-1):
        train[:, i] = (train[:, i] - np.mean(train[:, i]))/ np.std(train[:, i])
        impute[:, i] = (impute[:, i] - np.mean(impute[:, i]))/ np.std(impute[:, i])
    print('训练集大小为：',train.shape)
    #print('训练集y大小为：',test.shape)
    print('插值集x大小为：',impute.shape)
    return train,impute



def train_data_split(train,random_seed,rate):
    """训练集划分"""
    np.random.seed(random_seed)
    np.random.shuffle(train)
    k = int(train.shape[0] * rate)
    train_data = train[:k]
    test_data = train[k:]
    print('训练数据集大小为：', train_data.shape)
    print('测试数据集大小为：', test_data.shape)
    return train_data,test_data



def concat_mask_metric(data):
    num = data.shape[0]
    zeros = np.zeros((num,1),dtype = float)
    data = np.concatenate((data,zeros),axis=1)
    return data



def split_series(data,num):
    num_len = data.shape[0]
    num //= 2
    lst = []
    for i in range(0, num_len-num + 1):
        lst.append(data[i:i+num, ])
    lst = np.array(lst)
    return lst



def split_impute_series(data, num):
    num_len = data.shape[0]
    num //= 2
    lst = []
    for i in range(0, num_len - num + 1, num):
        lst.append(data[i:i + num, ])
    lst = np.array(lst)
    return lst



def Data_prepare(train,test,impute,batch_size = 64):
    """划分数据进入模型"""
    train_x = train[:,:-1]
    train_y = train[:,-1]
    train_x = concat_mask_metric(train_x)
    test_x = test[:,:-1]
    test_y = test[:,-1]
    test_x = concat_mask_metric(test_x)
    impute = concat_mask_metric(impute)
    n = impute.shape[0]
    train_x = split_series(train_x, n)
    train_y = split_series(train_y, n)
    test_x = split_series(test_x, n)
    test_y = split_series(test_y, n)
    impute = split_impute_series(impute,n)

    print('**** prepare data ****')
    print("训练集x大小为：",train_x.shape)
    print("训练集y大小为：", train_y.shape)
    print("测试集x大小为：", test_x.shape)
    print("测试集y大小为：", test_x.shape)
    print("插补集z大小为：", impute.shape)

    train_data = Data(train_x, train_y)
    test_data = Data(test_x, test_y)
    impute_data = Data2(impute)
    trainset = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    testset = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    imputeset = DataLoader(impute_data, batch_size=batch_size, shuffle=False)
    return trainset,testset,imputeset




if __name__ == '__main__':
    path = 'number'
    random_seed = 2023
    train,impute = load(path)
    train_data,test_data = train_data_split(train,random_seed,rate=0.7)
    trainset,testset,imputeset = Data_prepare(train_data,test_data,impute,batch_size=32)

