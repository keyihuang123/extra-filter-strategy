import pandas as pd
import numpy as np

from extraction import tatq_feature,kats_feature,talib_feature
from feature.lstm.LSTM import lstm_feature
from feature.hgnn.run_WPDP import hgnn_feature
from feature.transformer.train import transformer_feature




def run(df:pd.DataFrame,df_cols:pd.DataFrame,**kwargs):

    df_temp = df.copy()
    df_cols = df_cols.copy()

    extraction = kwargs['extraction']
    label_name = kwargs['label']['selection']['name']
    data_split = kwargs['data_split']['selection']
    contract   = kwargs['contract']['selection']['code']

    # 计算划分训练集和测试集的索引
    if data_split['method'] == 'rate':
        split_idx = df_temp.index[int(len(df_temp) * data_split['parameter'])]
    else:
        if data_split['parameter'] is None:
            split_idx = df_temp.index[int(len(df_temp) * 0.8)]
        else:
            split_idx =data_split['parameter']

    # 需要计算的特征
    opretion = extraction['operation']
    if opretion==None:
        return df_temp, df_cols, split_idx


    for step in opretion:
        method = opretion[step]['method']
        from_  = opretion[step]['from']
        params = opretion[step]['parameter']

        feature,df_cols_temp = None,None

        cols_name = df_cols[df_cols['type'] == from_[0]].name.tolist()  # 默认选择第一组特征

        if params is None:
            params = extraction['support'][method]['parameter']

        if method=='ta_tq':
            feature, df_cols_temp = tatq_feature(df_temp[cols_name],params,type_=step)

            # 天勤的ta计算时可能出现无穷值
            feature = feature.replace([np.inf, -np.inf], np.nan)

        elif method=='ta_lib':
            feature, df_cols_temp = talib_feature(df_temp[cols_name], params, type_=step)

        elif method=='kats':
            feature, df_cols_temp = kats_feature(df_temp[cols_name],column=params['column'],freq=params['freq'],
                                                 selected_features=params['selected_features'],contract=contract,type_=step)
            # 可能出现无穷值
            feature = feature.replace([np.inf, -np.inf], np.nan)


        elif method=='lstm':

            feature, df_cols_temp = lstm_feature(df_temp[cols_name], df_temp[label_name], split_idx, type_=step)

        elif method=='hgnn':
            cols_group = [df_cols[df_cols['type'] == i].name.tolist() for i in from_[:3]]

            feature, df_cols_temp = hgnn_feature(df_temp,cols_group,label_name,split_idx,type_=step,**params)

        elif method=='transformer':
            X,Y = df_temp[cols_name],df_temp[label_name]

            feature, df_cols_temp = transformer_feature(X,Y,split_idx,type_=step,**params)


        elif method=='combination':
            cond = df_cols['type'].apply(lambda x: True if x in from_ else False)
            cols_name = df_cols[cond].name.tolist()

            df_cols_temp = pd.DataFrame([{'type': step, 'name': col} for col in cols_name])

        else:
            pass

        if not feature is None:
            df_temp = df_temp.join(feature)
            df_temp.dropna(inplace=True)

        if not df_cols_temp is None:
            df_cols = df_cols.append(df_cols_temp, ignore_index=True)


    return df_temp,df_cols,split_idx
