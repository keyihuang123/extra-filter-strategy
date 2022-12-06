import pandas as pd
import numpy as np
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=60, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=65, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(in_features=65, out_features=16)
        self.dense2 = nn.Linear(in_features=16, out_features=2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x    = self.dropout(x)
        x, _ = self.lstm2(x)
        x    = self.dropout(x)

        e_feature = self.dense1(x[:, -1, :])
        x = self.dense2(e_feature)

        # softmax
        x = nn.functional.softmax(x, dim=-1)

        return x, e_feature



def lstm_feature(X,Y,split_dt,epochs:int=50 ,batch_size:int=32,type_:str=''):

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device("cpu") # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LSTM(X.shape[1]).to(device)
    ce_loss = nn.CrossEntropyLoss()

    epochs = epochs
    batch_size = batch_size
    lr = 1e-3
    # lr_ = torch.optim.lr_scheduler.StepLR(model.optim)

    X_ = pd.Series(X.rolling(10) ,index=X.index).apply(lambda x :x.values.tolist() if len(x)==10 else np.nan).dropna()
    Y_ = pd.Series(Y.rolling(10) ,index=Y.index).apply(lambda x :x.iloc[-1] if len(x)==10 else np.nan).dropna().astype(int)

    xtrain,ytrain = X_.loc[:split_dt],Y_.loc[:split_dt]
    xtest         = X_.loc[split_dt:]

    for epoch in range(epochs):
        start_idx = 0  # 开始训练的样本索引
        while 1:
            # 一个epoch训练完的条件
            if start_idx >len(xtrain):
                break

            end_idx = start_idx + batch_size

            x = torch.tensor(xtrain.iloc[start_idx:end_idx].tolist()).to(device)
            y = torch.tensor(ytrain.iloc[start_idx:end_idx].tolist()).to(device)

            ####################### forward #########################
            ypre ,_ = model(x)

            # 每个 batch_size 的loss
            loss = ce_loss(ypre ,y)

            ####################### backward #########################
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

            # 下一个 batch_size 的索引
            start_idx = end_idx

    # 返回提取的特征
    _ ,feature_train = model(torch.tensor(xtrain.values.tolist()).to(device))
    _, feature_test  = model(torch.tensor(xtest.values.tolist()).to(device))

    xtrain = pd.DataFrame(feature_train.cpu().detach().numpy() ,index=xtrain.index)
    xtest  = pd.DataFrame(feature_test.cpu().detach().numpy() ,index=xtest.index)

    X_ = xtrain.append(xtest)
    X_.columns = ['lstm_'+str(i) for i in X_.columns]
    df_cols = pd.DataFrame([{'type': type_, 'name': col} for col in X_.columns])

    return  X_,df_cols
