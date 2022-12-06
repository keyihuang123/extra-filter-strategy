'''
    工具模块
'''

import os
from yaml import dump,load
import joblib
from time import time
import random

def save_results(test_report,test_auc,test_acc,test_pred,config:dict,name:str='',dt:str=''):
    '''
    :param test_report: DataFrame,
    :param test_auc: DataFrame,
    :param test_acc: DataFrame,
    :param config: dict,
    :param name: 此次实验的方案名,
    :param dt: 测试集时间,
    :param kwargs: 保留兼容性
    :return: None
    '''

    if name=='':
        random.seed(int(time()))
        name = ''.join(random.sample('qwertyuioplkjhgfdsazxcvbnm',10))

    if dt=='':
        dt = str( int(time()) )

    src = os.path.join(os.getcwd(), '../results', name,dt)

    os.makedirs( src )

    #保存所有模型的得分
    test_report.to_csv(os.path.join(src,'test_report.csv'),index=False,float_format='%.4f')
    test_auc.to_csv(os.path.join(src,'test_auc.csv'),index=False, float_format='%.4f')
    test_acc.to_csv(os.path.join(src, 'test_acc.csv'), index=False, float_format='%.4f')
    test_pred.to_csv(os.path.join(src, 'test_pred.csv'))

    #保存配置
    with open(os.path.join(src, 'config.yaml'), 'w', encoding='utf8') as f:
        dump(config, f,sort_keys=False,allow_unicode=True)

