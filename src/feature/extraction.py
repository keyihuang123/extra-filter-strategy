'''
    特征工程
'''
import pandas as pd
import numpy as np
import ta_lib.ta as ta
from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
from copy import deepcopy

import talib

def talib_feature(df:pd.DataFrame,ta_params:dict,type_:str='',**kwargs):

    klines = df.copy()
    # 传入的klines数据列名有后缀，如: open_step0_RB00_5M,  将其改为基础的open,high,low,close,volume
    klines.columns = [i.split('_')[0] for i in klines.columns]

    df_temp = pd.DataFrame(index=klines.index)

    for item in ta_params:

        params = ta_params[item]['parameter']
        if params is None:
            params = {}

        columns = ta_params[item]['columns']
        data = [klines[col] for col in columns]

        res = getattr(talib,item)(*data,**params)

        if type(res)==tuple: # 指标返回多个输出
            for i in range(len(res)):
                df_temp[item+'_'+str(i)+'_'+type_] = res[i]
        else:
            df_temp[item+'_'+type_] = res

    df_cols = pd.DataFrame([{'type':type_,'name':col} for col in df_temp.columns])

    return df_temp,df_cols



def tatq_feature(df:pd.DataFrame,ta_params:dict,type_:str=''):
    '''
    此部分特征是以基础特征生成的 ，包含 高开低收，成交量
    字段名要求为： high,open,low,close,volume

    :param df:  DataFrame,K线数据
    :param type_: 对当次计算的特征进行归类
     :param ta_params: 需要计算的ta指标及参数，格式：{'ATR':{'n': 6,'name':xxx},'MA':{'n':6,'name':xxx},...}
    :return: df,df_col
    '''

    klines = df.copy()
    klines.index = range(len(klines))

    # 传入的klines数据列名有后缀，如: open_step0_RB00_5M,  将其改为基础的open,high,low,close,volume
    klines.columns = [i.split('_')[0] for i in klines.columns]

    ta_params = deepcopy(ta_params)

    cols = []

    for ta_ in ta_params:
        conf = ta_params[ta_]
        conf['df'] = klines

        del conf['name']

        df_temp = getattr(ta, ta_)(**conf)
        df_temp.columns = [i + '_' + ta_ + '_' +type_ for i in df_temp.columns]

        cols += df_temp.columns.tolist()
        klines = klines.join(df_temp)

    klines = klines[cols]
    klines.index = df.index

    df_cols = pd.DataFrame( [{'type': type_, 'name': col} for col in cols] )

    return klines,df_cols


def kats_feature(df:pd.DataFrame,column:str='close',freq:int=100,selected_features=None,contract='',type_:str=''):
    '''

    :param df: DataFrame,必须包含两列数据：time,close, 其中 time为 datetime类型
    :param column: str, 用哪个特征计算kats
    :param freq: int, 周期
    :param type_: str, 计算特征所属的类型
    :return: data,df_col => DataFrame,DataFrame
    '''

    df_kats = pd.read_csv('../data/{}_close_kats_100.csv'.format(contract),index_col=0)
    df_kats.index = pd.to_datetime(df_kats.index)
    df_kats.columns = [col + '_' + type_ for col in df_kats.columns]
    df_cols = pd.DataFrame([{'type': type_, 'name': col} for col in df_kats.columns])


    '''
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

    #df_kats.columns = [col + '_' + type_ for col in df_kats.columns]
    df_cols = pd.DataFrame([{'type': type_, 'name': col} for col in df_kats.columns])

    '''

    '''
            f_col结构：Series

            0       nan
            1       nan
            2       nan


            1233    {'mean':xxx,'var':xxx,...}
            1234    {'mean':xxx,'var':xxx,...}
            1235    {'mean':xxx,'var':xxx,...}

    '''


    return df_kats ,df_cols

if __name__=='__main__':

    import warnings
    warnings.filterwarnings('ignore')

    import yaml
    config = None
    with open('../../config.yaml', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    params = config['extraction']['support']['kats']['parameter']

    klines = pd.read_csv('../../data/{}_复权.csv'.format('M00_5M')) # ,index_col=0
    #klines['datetime'] = pd.to_datetime(klines.datetime.astype('int64'), format='%Y%m%d%H%M%S')
    print(klines)
    klines['datetime'] = pd.to_datetime(klines['datetime'])

    # 将datetime列设置为索引，会将datetime从数据中删除
    klines.set_index('datetime', inplace=True)

    # 按时间 筛选数据
    klines = klines[['high', 'open', 'low', 'close', 'volume']]

    #print(params['selected_features'])
    df,_ = kats_feature(klines,selected_features=params['selected_features'])

    df.to_csv('../../data/m00_5m_close_kats_100.csv')