"""
Funções para separação do conjunto de dados em treino e teste.
"""


def separacao_dados_treino_teste(df, target_col, sep_type, features_cols=[],
                                 cat_cont_var_bins=5, test_size=0.2,
                                 random_state=10):

    """
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split

    if (test_size <= 0) or (test_size >= 0.5):
        raise Exception('A fração da amostra de teste deve ser maior que' +
                        ' 0.0 e menor que 0.5.')

    if ((sep_type != 'classification') and
       (sep_type != 'regression') and
       (sep_type != 'time_series')):
        raise Exception('Valor inválido no parâmetro sep_type ' +
                        '(somente classification, regression e time_series ' +
                        'são permitidos)')

    if len(features_cols) == 0:
        features_cols = list(df.drop(columns=target_col).columns)

    df_temp = df[[target_col] + features_cols].copy()

    if sep_type == 'classification':

        train, test = train_test_split(df_temp,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=True,
                                       stratify=df_temp[target_col])

    elif sep_type == 'regression':

        labels_cat = []
        for i in range(cat_cont_var_bins):
            labels_cat.append(i)
        df_temp.loc[:, 'categorical_target'] = pd.cut(df_temp[target_col],
                                                      bins=cat_cont_var_bins,
                                                      labels=labels_cat)
        train, test = train_test_split(df_temp,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=True,
                                       stratify=df_temp['categorical_target'])

        train = train[[target_col] + features_cols]
        test = test[[target_col] + features_cols]

    elif sep_type == 'time_series':

        index_type = str(df_temp.index.dtype)
        if 'datetime' not in index_type:
            raise Exception('O indice do DataFrame deve estar no ' +
                            'formato datetime.')

        df_temp.sort_index(inplace=True)

        train_size = int(df_temp.shape[0] * (1.-test_size))
        train = df_temp.iloc[:train_size, :]
        test = df_temp.iloc[train_size:, :]

    X_train = train[features_cols].copy()
    y_train = train[[target_col]].copy()
    X_test = test[features_cols].copy()
    y_test = test[[target_col]].copy()

    return X_train, y_train, X_test, y_test
