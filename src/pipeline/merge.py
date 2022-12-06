'''

    1、各种数据源整合
    2、样本标签的制作

'''
import pandas as pd
import pipeline.label as lb

# 获取基础K线数据
def get_klines(contract:str,start_dt:str,end_dt:str):
    '''

    :param contract: str, 合约代码：RB00_5M,I00_5M
    :param start_dt: str,开始时间 2022-01-01
    :param end_dt: str, 结束时间 2022-06-01
    :param type_: str,  是基础数据的K线还是辅助数据的K线, 基础数据 base
    :return: data,columns  => DataFrame,DataFrame

    df_cols:
                type                  name
        0  step0_RB00_5M    high_step0_RB00_5M
        1  step0_RB00_5M    open_step0_RB00_5M
        2  step0_RB00_5M     low_step0_RB00_5M
        3  step0_RB00_5M   close_step0_RB00_5M
        4  step0_RB00_5M  volume_step0_RB00_5M
    '''

    klines = pd.read_csv('../data/{}_复权.csv'.format(contract)) # ,index_col = 0
    #klines['datetime'] =  pd.to_datetime(klines.datetime.astype('int64'), format='%Y%m%d%H%M%S')
    klines['datetime'] = pd.to_datetime(klines['datetime'])

    # 将datetime列设置为索引，会将datetime从数据中删除
    klines.set_index('datetime', inplace=True)

    # 按时间 筛选数据
    klines = klines[['high', 'open', 'low', 'close', 'volume']].loc[start_dt:end_dt]


    klines.columns = [i+'_step0_'+contract  for i in klines.columns]

    # 记录基础k线数据部分包含的特征列
    df_cols = pd.DataFrame( [{'type': 'step0_'+contract, 'name': col} for col in klines.columns] )

    return klines,df_cols


# 合并数据
def run(**kwargs):

    contract = kwargs['contract']['selection']
    label = kwargs['label']['selection']

    # 保存各种类别特征的列名
    df_cols = pd.DataFrame(columns=['type','name'])

    # 基础K线数据
    klines,df_cols_temp = get_klines(contract=contract['code'],start_dt=contract['start_dt'],end_dt=contract['end_dt'])
    df_cols = df_cols.append(df_cols_temp,ignore_index=True)

    ### 添加辅助数据 ###

    # 获取完基础K先后 制作标签
    df_label = getattr(lb,label['name'])(klines,**label['parameter'])

    # 合并数据，默认以 索引 为连接键
    data = df_label.join(klines,how='left')

    # 删除产生的缺失值样本，修改标签类型
    data.dropna(inplace=True)

    return data,df_cols
