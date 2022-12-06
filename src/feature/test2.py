import pandas as pd
import numpy as np

from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData


def kats_feature(df:pd.DataFrame,column:str='close',freq:int=100,selected_features=None,type_:str=''):
    '''

    :param df: DataFrame,必须包含两列数据：time,close, 其中 time为 datetime类型
    :param column: str, 用哪个特征计算kats
    :param freq: int, 周期
    :param type_: str, 计算特征所属的类型
    :return: data,df_col => DataFrame,DataFrame
    '''

    #df_kats = pd.read_csv('../data/rb00_5m_close_kats_100.csv',index_col=0)
    #df_kats.index = pd.to_datetime(df_kats.index)
    #df_kats.columns = [col + '_' + type_ for col in df_kats.columns]
    #df_cols = pd.DataFrame([{'type': type_, 'name': col} for col in df_kats.columns])



    klines = df.copy()
    # 传入的klines数据列名有后缀，如: open_step0_RB00_5M,  将其改为基础的open,high,low,close,volume
    klines.columns = [col.split('_')[0] for col in klines.columns]

    klines['time'] = klines.index
    klines = klines[['time', column]]

    if selected_features is None:
        ts = TsFeatures()
    else:
        ts = TsFeatures(selected_features=selected_features)

    f_col = pd.Series(klines.rolling(freq)) \
        .apply(lambda x: ts.transform(TimeSeriesData(x)) if len(x) == freq else np.nan)

    f_col.index = klines.index
    f_col.dropna(inplace=True)

    df_kats = pd.DataFrame(f_col.tolist(), index=f_col.index)

    df_cols = pd.DataFrame([{'type': type_, 'name': col} for col in df_kats.columns])



    return df_kats ,df_cols


import warnings
warnings.filterwarnings('ignore')

import yaml
config = None
with open('../../config.yaml', encoding='utf8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

params = config['extraction']['support']['kats']['parameter']

klines = pd.read_csv('../../data/{}_复权.csv'.format('HSI00_5M'))
print(klines)
klines['datetime'] = pd.to_datetime(klines['datetime'])

# 将datetime列设置为索引，会将datetime从数据中删除
klines.set_index('datetime', inplace=True)

# 按时间 筛选数据
klines = klines[['high', 'open', 'low', 'close', 'volume']]

#print(params['selected_features'])
df,_ = kats_feature(klines,selected_features=params['selected_features'])

df.to_csv('../../data/HSI00_5M_close_kats_100.csv')