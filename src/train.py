from pipeline.merge import run as mrun
from feature.engineering import run as frun
from models.classification import run as crun
from pipeline.preprocessing import normalize
from copy import deepcopy


def fit(**kwargs):
    config = deepcopy(kwargs)
    label_name = config['label']['selection']['name']

    print('开始整合数据...')
    df,df_cols = mrun(**config)
    print('完成')


    print('开始特征工程...')
    df,df_cols ,split_idx = frun(df,df_cols,**config)
    print('完成')

    print('开始训练模型...')

    # 使用何种特征进行训练 用最后一个step得到的特征进行训练
    train_type = df_cols['type'].iloc[-1]
    cols       = []
    if train_type.startswith('step0'):  # 只用基础K线数据
        cols = df_cols['name'].tolist()
    else:
        cols = df_cols[df_cols['type']==train_type].name.tolist()

    # 标准化归一化操作
    df[cols] = normalize(df[cols], split_idx, **kwargs)  # 有缺失值不影响标准化


    # 划分训练集和测试集
    X,y = df.loc[:split_idx,cols],df.loc[:split_idx,label_name]
    xtest,ytest = df.loc[split_idx:,cols],df.loc[split_idx:,label_name]

    print('完成')
    return crun(X,y,xtest,ytest,**config) # test_report,test_auc,test_acc,test_pred


if __name__=='__main__':
    import yaml
    import numpy as np
    import random
    import warnings
    warnings.filterwarnings('ignore')

    from feature.transformer.train import transformer_feature

    random.seed(0)
    np.random.seed(0)

    config = {}
    with open('../config.yaml', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    fit(**config)
    #X = df.iloc[:,1:]
    #y = df.iloc[:,0]


    #test_report,test_auc,test_acc = fit(**config)
    #print(test_auc)
    #print(test_acc)
    #print(test_report)
