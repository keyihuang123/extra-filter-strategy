import numpy as np
import pandas as pd

def stop_surplus(klines: pd.DataFrame, delta: float = 0.01,name:str='stop_surplus'):
    '''

    止盈标签，分类：
    存在某一天涨幅超过 deltal 为 1 ， 跌幅超过delta为 0

    :param klines: 基础K线数据
    :param delta:  阈值
    :return: DataFrame，只有一列
    '''
    # 传入的klines数据列名有后缀，如: open_step0_RB00_5M,  将其改为基础的open,high,low,close,volume
    klines = klines.copy()
    klines.columns = [i.split('_')[0] for i in klines.columns]

    close = klines[['close']]
    close[name] = np.nan

    for i in range(len(close)-1):
        tr = close.close.iloc[i+1:] / close.close.iloc[i] - 1

        for j in tr:
            if j >= delta:
                close[name].iloc[i] = 1
                break
            if j <= -delta:
                close[name].iloc[i] = 0
                break

    return close[[name]]


def stop_surplus_limit(klines: pd.DataFrame, delta: float = 0.01, n: int = 12,name:str='stop_surplus_limit'):
    '''

    止盈 分类标签 ， 限制条件
    未来 n条 K线中，首先出现涨幅超过 delta 的设置为买入，出现跌幅超过 delta 的卖出，其余情况设置为不操作 0。

    :param klines: DataFrame, k线数据
    :param delta:  阈值
    :param n:      限制的K线数，即在未来多少条k线内达到 delta 的条件
    :return: DataFrame
    '''

    # 传入的klines数据列名有后缀，如: open_step0_RB00_5M,  将其改为基础的open,high,low,close,volume
    klines = klines.copy()
    klines.columns = [i.split('_')[0] for i in klines.columns]

    close = klines[['close']].copy()
    close[name] = np.nan

    for i in range(len(close) - n):
        x = close.close.iloc[i:i + n]
        x = x / x.iloc[0] - 1

        if x.max() <= delta and x.min() >= -delta:
            close[name].iloc[i] = 0
            continue

        for j in x:
            if j > delta:
                close[name].iloc[i] = 1
                break
            if j < -delta:
                close[name].iloc[i] = -1
                break

    return close[[name]]
