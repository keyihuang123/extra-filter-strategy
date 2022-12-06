import pandas as pd
import numpy as np
from imblearn.over_sampling import SVMSMOTE
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import TimeSeriesSplit

from sklearn import metrics

class Models:
    def __init__(self):
        '''
        所有模型的实例都在此类中定义
        '''
        pass

    def logistic(self):
        return LogisticRegression(random_state=0)

    def knn(self):
        return KNeighborsClassifier(10)

    def nb(self):
        return GaussianNB()

    def xgboost(self):
        return XGBClassifier(random_state=0,n_jobs=-1)

    def catboost(self):
        return CatBoostClassifier(random_state=0,verbose=False)

    def lightgbm(self):
        return LGBMClassifier(max_depth=10,random_state=0,n_jobs=-1)


def report(y_true,y_pre,model_name:str=''):
    '''

    返回结果示例：
        class   precision   recall      f1      support       model
          0         0.55      0.55      0.55      1000      logistic
          1         0.55      0.55      0.55      1000      logistic


    :param y_true:  真实标签
    :param y_pre:   预测标签
    :param model_name: 对应的模型名
    :return: pd.DataFrame(columns=[class,precision,recall,f1,support,model])
    '''

    df_temp = pd.DataFrame(columns=['class','precision','recall','f1-score','support','model'])
    reports = metrics.classification_report(y_true, y_pre, output_dict=True)

    for l in np.unique(y_true):
        score = reports.get(str(l))
        if score == None:
            score = {'precision':0,'recall':0,'f1-score':0,'support':0}
        score['class'] = l
        score['model'] = model_name
        df_temp = df_temp.append([score],ignore_index=True)

    return df_temp


def run(X,y,xtest,ytest,**kwargs):

    balance = kwargs.get('balance')
    models_name = kwargs.get('models_name')

    X = X.copy()
    y = y.copy()

    models = Models()
    test_report = pd.DataFrame() # 记录 prec. recall  f1
    test_auc    = pd.DataFrame() # 记录 auc
    test_acc    = pd.DataFrame() # 记录 acc
    test_pred   = pd.DataFrame(ytest.tolist(), columns=['label'], index=ytest.index) # 保存测试集的预测结果

    for name in models_name:
        model = getattr(models,name)()

        if balance:
            X, y = SVMSMOTE(random_state=0,n_jobs=-1).fit_resample(X, y)

        model.fit(X,y)

        ypre = model.predict(xtest)            # 预测标签，输出：array([0,1,0,0,...])
        ypre_prob = model.predict_proba(xtest) # 预测概率，输出：array([[0.2,0.8],[0.3,0.7],...])

        test_report = test_report.append(report(ytest, ypre, name), ignore_index=True)
        test_acc = test_acc.append([{'model': name, 'acc': metrics.accuracy_score(ytest, ypre)}],ignore_index=True)
        test_auc = test_auc.append([{'model':name,'auc':metrics.roc_auc_score(ytest,ypre_prob[:,1])}] ,ignore_index=True)

        test_pred[name] = pd.Series(ypre, index=ytest.index)


    return test_report,test_auc,test_acc,test_pred
