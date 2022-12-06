import pandas as pd
import numpy as np
import yaml
import random

from utils import save_results
from train import fit

import warnings
warnings.filterwarnings('ignore')

random.seed(0)
np.random.seed(0)

############################################
'''
import logging
logging.disable(logging.CRITICAL)

logging.getLogger('pycaret')
logging.basicConfig(
        filename= os.path.join( os.getcwd(),'log','logs.log' ) ,
        filemode='w',
        level=logging.ERROR,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
'''
############################################

config = None
with open('../config.yaml', encoding='utf8') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)


from datetime import datetime
from datetime import timedelta

# 13个月时间训练
start_dt = pd.date_range('2016-11-01','2021-04-01',freq='2M')
split_dt = pd.date_range('2017-12-01','2022-05-01',freq='2M')
end_dt = pd.date_range('2018-02-01','2022-07-01',freq='2M')


# 半年训练
#start_dt = pd.date_range('2020-05-01','2022-01-01',freq='2M')
#split_dt = pd.date_range('2020-10-01','2022-05-01',freq='2M')
#end_dt = pd.date_range('2020-12-01','2022-07-01',freq='2M')


start_dt = [str(datetime.strptime(str(i),'%Y-%m-%d %H:%M:%S') + timedelta(days=1)) for i in start_dt]
split_dt = [str(datetime.strptime(str(i),'%Y-%m-%d %H:%M:%S') + timedelta(days=1)) for i in split_dt]
end_dt = [str(datetime.strptime(str(i),'%Y-%m-%d %H:%M:%S') + timedelta(days=1)) for i in end_dt]


# 18年1月1 到22年6月28 4年半的表现
name = 'lstm(base_tatq_kats)_cl00'   # 为每次实验命名

for i,j,k in zip(start_dt,split_dt,end_dt):
    config['contract']['selection']['start_dt'] = i
    config['contract']['selection']['end_dt'] = k
    config['data_split']['selection']['parameter'] = j

    test_report, test_auc,test_acc,test_pred = fit(**config)

    save_results(test_report, test_auc,test_acc,test_pred ,config, name=name,dt=str(k).split(' ')[0])


## 探索模型
#test_report,test_auc ,test_acc =  fit(**config)

## 保存探索记录
#save_results(test_acc,test_report,test_score_prob,config,name='',dt='')

## 探索完毕，选择一个模型及配置保存，用于回测及实时预测
#save_model(config=config,model_name='lr',name='演示')
