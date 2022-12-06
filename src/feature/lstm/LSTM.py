import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim


class LSTM(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=60, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=65, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(in_features=65, out_features=16)
        self.dense2 = nn.Linear(in_features=16, out_features=2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-2)  # -1 timestep  -2 features -3 samples
        x, _ = self.lstm1(x)
        x    = self.dropout(x)
        x, _ = self.lstm2(x)
        x    = self.dropout(x)

        e_feature = self.dense1(x[:, -1, :])
        x = self.dense2(e_feature)

        # softmax
        x = nn.functional.softmax(x, dim=-1)

        return x, e_feature


class LSTM_fix(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        # num_layers: 堆叠多少个LSTM，设为2即和原本的一样。 Default: 1
        # dropout: 在每个LSTM层引入一个dropout层，不需要额外写. Default: 0
        # bidirectional: 双向LSTM，可以提升，但是输出维度*2. Default: ``False``
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=3, batch_first=True,
                             bidirectional=True, dropout=0.1)
        # 因为双向LSTM，输出维度in_features *2
        self.dense1 = nn.Linear(in_features=input_size * 2, out_features=16)
        self.dense2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-2)  # -1 timestep  -2 features -3 samples
        x, _ = self.lstm1(x)

        e_feature = self.dense1(x[:, -1, :])
        x = self.dense2(e_feature)

        # softmax
        x = nn.functional.softmax(x, dim=-1)

        return x, e_feature


def train(X,y,batch_size,model,loss_fn, optimizer,device):
    model.train()
    start_idx = 0
    samples_num = len(X)
    f = []

    while start_idx < samples_num:

        end_idx = start_idx + batch_size
        x_ = torch.tensor(X.iloc[start_idx:end_idx].tolist()).to(device)
        y_ = torch.tensor(y.iloc[start_idx:end_idx].tolist()).to(device)

        ypred , feature = model(x_)
        f.append( feature.cpu().detach().numpy().tolist())

        loss = loss_fn(ypred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if start_idx % 100 == 0:
            loss, current = loss.item(), start_idx
            print(f"loss: {loss:>7f}  [{current:>5d}/{samples_num:>5d}]")

        # 下一个 batch_size 开始的索引
        start_idx = end_idx

    return f

def test(X,y,batch_size,model,loss_fn,device):
    model.eval()
    start_idx = 0
    samples_num = len(X) # 测试集样本数量
    batches_num = 0  # 训练了多少个batch
    test_loss, correct = 0, 0
    f = []
    with torch.no_grad():
        while start_idx < samples_num:

            end_idx = start_idx + batch_size
            x_ = torch.tensor(X.iloc[start_idx:end_idx].tolist()).to(device)
            y_ = torch.tensor(y.iloc[start_idx:end_idx].tolist()).to(device)

            ypred, feature = model(x_)
            f.append(feature.cpu().detach().numpy().tolist())

            test_loss += loss_fn(ypred, y_).item()
            correct += (ypred.argmax(1) == y_).type(torch.float).sum().item()

            # 下一个 batch_size 开始的索引
            start_idx = end_idx
            batches_num += 1

    test_loss /= batches_num
    correct /= samples_num
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return f



def train2(X,y,batch_size,model,loss_fn, optimizer,device,seq_len=10,verbose=False):
    model.train()
    start_idx = seq_len-1    # 从第 seq_len 个开始
    samples_num = len(y)     # 样本数量
    feature_num = X.shape[1]


    while start_idx < samples_num:
        end_idx = start_idx + batch_size

        if end_idx>=samples_num:
            end_idx = samples_num

        x_ = []
        y_ = []
        for i in range(start_idx,end_idx):
            x_ += X.iloc[i-seq_len+1:i+1,:].values.tolist()
            y_ += [y.iloc[i]]

        x_ = torch.tensor(np.array(x_).reshape((-1,seq_len,feature_num)),dtype=torch.float32).to(device)
        y_ = torch.tensor(np.array(y_).reshape((-1,)),dtype=torch.long).to(device)


        ypred , _ = model(x_)
        loss = loss_fn(ypred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            if (start_idx-seq_len+1) % 100 == 0:
                loss, current = loss.item(), start_idx
                print(f"loss: {loss:>7f}  [{current:>5d}/{samples_num:>5d}]")

        # 下一个 batch_size 开始的索引
        start_idx = end_idx


def test2(X,y,batch_size,model,loss_fn,device,seq_len=10,verbose=False):
    model.eval()
    start_idx = seq_len-1
    samples_num = len(y)
    feature_num = X.shape[1]

    batches_num = 0      # 训练了多少个batch
    test_loss, correct = 0, 0
    ypred_list = []

    with torch.no_grad():
        while start_idx < samples_num:

            end_idx = start_idx + batch_size

            if end_idx >= samples_num:
                end_idx = samples_num

            x_ = []
            y_ = []
            for i in range(start_idx,end_idx):
                x_ += X.iloc[i - seq_len + 1:i + 1, :].values.tolist()
                y_ += [y.iloc[i]]

            x_ = torch.tensor(np.array(x_).reshape((-1, seq_len, feature_num)),dtype=torch.float32).to(device)
            y_ = torch.tensor(np.array(y_).reshape((-1,)),dtype=torch.long).to(device)

            ypred, feature = model(x_)
            ypred_list += ypred.cpu().detach().numpy().tolist()

            test_loss += loss_fn(ypred, y_).item()
            correct += (ypred.argmax(1) == y_).type(torch.float).sum().item()

            # 下一个 batch_size 开始的索引
            start_idx = end_idx
            batches_num += 1

    test_loss /= batches_num
    correct /= samples_num

    if verbose:
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return np.array(ypred_list)

def get_feature(X,model,device,batch_size=32,seq_len=10):
    model.eval()
    model = model.to(device)

    f = []

    start_idx = seq_len-1
    samples_num = len(X)
    feature_num = X.shape[1]

    with torch.no_grad():
        while start_idx < samples_num:

            end_idx = start_idx + batch_size

            if end_idx >= samples_num:
                end_idx = samples_num

            x_ = []
            for i in range(start_idx, end_idx):
                x_ += X.iloc[i - seq_len + 1:i+1,:].values.tolist()

            x_ = torch.tensor(np.array(x_).reshape((-1, seq_len, feature_num)), dtype=torch.float32).to(device)

            _, feature = model(x_)
            f += feature.cpu().detach().numpy().tolist()  # feature.shape => (batch_size,16)

            # 下一个 batch_size 开始的索引
            start_idx = end_idx


    f = pd.DataFrame(f,index=X.index[seq_len-1:])

    return f

def lstm_feature(X,Y,split_dt,epochs:int=3,batch_size:int=18130,type_:str=''):

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device('cuda:0' if not torch.cuda.is_available() else 'cpu')
    model = LSTM_fix(X.shape[1]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)


    X_ = pd.Series(X.rolling(10) ,index=X.index).apply(lambda x :x.values.tolist() if len(x)==10 else np.nan).dropna()
    Y_ = pd.Series(Y.rolling(10) ,index=Y.index).apply(lambda x :x.iloc[-1] if len(x)==10 else np.nan).dropna().astype(int)

    xtrain,ytrain = X_.loc[:split_dt],Y_.loc[:split_dt]
    xtest,ytest   = X_.loc[split_dt:],Y_.loc[split_dt:]

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}\n-------------------------------")

        feature_train = train(xtrain,ytrain,batch_size,model,loss_fn, optimizer,device)
        feature_test = test(xtest,ytest,batch_size,model,loss_fn,device)

    # 返回提取的特征

    _ ,feature_train = model(torch.tensor(xtrain.values.tolist()).to(device))
    ytest, feature_test  = model(torch.tensor(xtest.values.tolist()).to(device))
   
    xtrain = pd.DataFrame(feature_train ,index=xtrain.index,dtype=np.float32)
    xtest  = pd.DataFrame(feature_test,index=xtest.index,dtype=np.float32)

    X_ = xtrain.append(xtest)
    X_.columns = [type_+'_'+str(i) for i in X_.columns]

    df_cols = pd.DataFrame([{'type': type_, 'name': col} for col in X_.columns])

    return  X_,df_cols
