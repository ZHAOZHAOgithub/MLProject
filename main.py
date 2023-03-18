from read_data import *
from IGNNK import *
from metrics import *
from Exp import  *
import argparse
import random
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

random_seed = 2023
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

parser = argparse.ArgumentParser(description='MLP for prediction or classification')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=2000, help='train epochs')
parser.add_argument('--train_epochs2', type=int, default=5000, help='train epochs for classification')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--batch_size2', type=int, default=256, help='batch size of train input data for for classification')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')#0.00005
parser.add_argument('--learning_rate2', type=float, default=0.000025, help='optimizer learning rate for classification')#0.00029
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# data loader

parser.add_argument('--path', type=str, default='number', help='number or label')
parser.add_argument('--data_path', type=str, default='../data/', help='data file')
parser.add_argument('--rate', type=str, default=0.7, help='rate of training')

# model

parser.add_argument('--time_step', type=int, default=25, help='time_step of series')# 25 or 10 序列长度
parser.add_argument('--hidden_size',type=int, default=150, help='hidden_size of DGCN')# DGCN的隐藏变量维度
parser.add_argument('--k', type=int, default=1, help='k step of DGCN')# DGCN的扩散卷积阶数
parser.add_argument('--num', type=int, default=9, help='number of series')# 9 or 13#序列总数

parser.add_argument('--time_step2', type=int, default=10, help='time_step of series')# 25 or 10
parser.add_argument('--hidden_size2',type=int, default=40, help='hidden_size of DGCN')
parser.add_argument('--k2', type=int, default=1, help='k step of DGCN')
parser.add_argument('--num2', type=int, default=13, help='number of series')# 9 or 13



args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
print('Whether the GPU of the computer has been started:   ',args.use_gpu)



if __name__ == '__main__':

    """路径准备"""

    args.path = 'label' # number or label

    """数据准备"""

    train,impute = load(args.path)
    train_data,test_data = train_data_split(train,random_seed,rate=args.rate)
    trainset,testset,imputeset = Data_prepare(train_data,test_data,impute,batch_size=args.batch_size)
    device = torch.device("cuda:0")
    """模型选择"""
    if args.path == 'number':
        model = IGNNK(args.time_step,args.hidden_size,args.k,args.num)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.MSELoss().to(device)
        mse = 10000
        for epoch in range(0, args.train_epochs):
            Train(model, trainset, criterion, optimizer)
            if epoch % 10 == 9:
                s = Test(model, testset)
                if s < mse:
                    mse = s
                    Impute(model,imputeset)
    elif args.path == 'label':
        model = IGNNK_label(args.time_step2,args.hidden_size2,args.k2,args.num2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate2)
        criterion = torch.nn.BCELoss().to(device)
        #criterion = torch.nn.BCEWithLogitsLoss()
        #criterion = nn.CrossEntropyLoss()
        acc = 0
        for epoch in range(0, args.train_epochs2):
            Label_Train(epoch,model, trainset, criterion, optimizer)
            if epoch % 10 == 9:
                s = Label_Test(model, testset,criterion)
                if s > acc:
                    acc = s
                    Label_Impute(model,imputeset)
