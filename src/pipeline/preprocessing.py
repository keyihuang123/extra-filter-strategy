from sklearn.preprocessing import StandardScaler,MinMaxScaler

def normalize(feature,split_idx,**kwargs):

    feature_ = feature.copy()

    method = kwargs['normalize']['selection']

    if method == 'zscore':
        m = StandardScaler()
        feature_.loc[:split_idx, :] = m.fit_transform(feature_.loc[:split_idx, :])
        feature_.loc[split_idx:, :] = m.transform(feature_.loc[split_idx:, :])

    elif method == 'minmax':
        m = MinMaxScaler()
        feature_.loc[:split_idx, :] = m.fit_transform(feature_.loc[:split_idx, :])
        feature_.loc[split_idx:, :] = m.transform(feature_.loc[split_idx:, :])

    else:
        pass

    return feature_
