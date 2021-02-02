def scaling(df, scaling_methods, kwargs=None):

    import pandas as pd
    from sklearn.pipeline import Pipeline

    list_scaler = []
    for i in range(len(scaling_methods)):

        scaling_method = scaling_methods[i]

        if scaling_method == 'min_max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler

        elif scaling_method == 'max_abs':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler

        elif scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler

        elif scaling_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler

        elif scaling_method == 'normalizer':
            from sklearn.preprocessing import Normalizer
            scaler = Normalizer

        elif scaling_method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer

        elif scaling_method == 'power_transform':
            from sklearn.preprocessing import PowerTransformer
            scaler = PowerTransformer

        else:
            raise Exception('Método inválido (' + scaling_method + ')')

        if (kwargs is None) or (kwargs[i] is None):
            scaler = scaler()
        else:
            scaler = scaler(kwargs[i])

        list_scaler.append((scaling_method, scaler))

    pipe = Pipeline(list_scaler)

    df_temp = pipe.fit_transform(df)
    df_temp = pd.DataFrame(data=df_temp, columns=df.columns)

    return df_temp, pipe
