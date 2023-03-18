from metrics import *
import argparse
import random
import numpy as np
import torch

device = torch.device("cuda:0")

def Train(model,train_loader,criterion,optimizer):
    model.train()
    loss_a = 0
    for x,y in train_loader:
        out = model(x.to(device))
        loss = criterion(out, y.to(device))
        loss_a += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()


def Test( model,test_loader):
    model.eval()
    count = 0
    Mae = 0
    Mse = 0
    Rmse = 0
    Mape = 0
    Mspe = 0
    Rse = 0

    for x, y in test_loader:
        out = model(x.to(device))
        count += x.shape[0]
        pred = out.detach().cpu().numpy()
        y = y.numpy()
        '''print(pred)
        print(y)'''
        mse, mae,  rmse, mape, mspe, rse = Num_metric(pred,y)
        Mae += mae
        Mse += mse
        Rmse += rmse
        Mape += mape
        Rse += rse
    print("||mse:  ", Mse / count, "mae:  ", Mae / count, "||rmse:  ", Rmse / count, "||mape:  ", Mape / count,
          "||mspe:  ", Mspe / count, "||rse:  ", Rse / count)
    return Mse/count
    #print("||mse:  ", Mse/count, "mae:  ", Mae/count,  "||rmse:  ", Rmse/count, "||mape:  ", Mape/count, "||mspe:  ", Mspe/count,  "||rse:  ", Rse/count )


def Impute(model,impute_loader):
    model.eval()
    print("打印插补结果")
    for x in impute_loader:
        out = model(x.to(device))
        out = list(out.detach().cpu().numpy())
        for i in out:
            for j in i:
                print(j)

def Label_Train(epoch,model,train_loader,criterion,optimizer):
    model.train()
    count = 0
    loss_a = 0
    for x,y in train_loader:
        #print(x.shape,y.shape)
        out = model(x.to(device))
        #print(out)
        count += x.shape[0]

        loss = criterion(out, y.to(device))

        weight = torch.zeros_like(y).float().cuda()
        weight = torch.fill_(weight, 0.7)
        weight[y > 0] = 0.3
        loss = torch.mean(weight * loss)

        loss_a += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()
    '''if epoch % 10 == 9:
        print(loss_a)'''

def Label_Test( model,test_loader,criterion):
    model.eval()
    count = 0
    Acc = 0
    Pr = 0
    Re = 0
    F1 = 0
    Pred = np.array([])
    Y = np.array([])
    loss_a = 0
    for x, y in test_loader:

        count += x.shape[0]
        out = model(x.to(device))
        count += x.shape[0]
        #m = torch.nn.Sigmoid()
        #out = m(out)
        '''loss = criterion(out, y)
        loss_a += loss.item()'''
        pred = out.detach().cpu().numpy()
        pred = np.where(pred > 0.5, 1, 0)
        #pred = pred.round()
        y = y.detach().cpu().numpy()
        '''print(Pred.shape,pred.shape)
        print(Y.shape, y.shape)'''
        pred = pred.flatten()
        y = y.flatten()
        Pred = np.concatenate((Pred,pred), axis=0)
        Y = np.concatenate((Y, y), axis=0)
    #print(loss_a/count)
    Pred = Pred.astype(int)
    Y = Y.astype(int)
    accuracy, precision, recall, F1 = Label_metric(Pred,Y)
    print("ACC:",accuracy, precision, recall, F1)
    return accuracy
    #print("||mse:  ", Mse/count, "mae:  ", Mae/count,  "||rmse:  ", Rmse/count, "||mape:  ", Mape/count, "||mspe:  ", Mspe/count,  "||rse:  ", Rse/count )


def Label_Impute(model,impute_loader):
    model.eval()
    print("打印插补结果")
    for x in impute_loader:
        out = model(x.to(device))
        out = list(out.detach().cpu().numpy().astype(int))
        for i in out:
            for j in i:
                print(j)