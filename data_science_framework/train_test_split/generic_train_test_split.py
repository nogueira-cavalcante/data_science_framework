# -*- coding: utf-8 -*-

"""Separação treino-teste."""


def generic_train_test_split(df, sep_type, target_col, features_cols=None,
                             cat_cont_var_bins=3, test_size=0.2,
                             lower_limit_date_train=None,
                             upper_limit_date_train=None,
                             lower_limit_date_test=None,
                             upper_limit_date_test=None, random_state=10):
    """
    Separação genérica do DataFrame de entrada em treino e teste.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo as features e o target.
    sep_type : string
        Tipo de separação treino-teste:
            - "classification": A separação treino-teste é estratificada
                                baseada no target.
            - "regression": É criada uma variável categórica temporária
                            baseada no target e no valor em cat_cont_var_bins.
                            Logo, a separação treino-teste é estratificada
                            baseada nessa variável categórica temporária.
           - "time-series": O índice do DataFrame de entrada precisa estar no
                            formato datetime. Logo, o DaFrame é ordenado pelo
                            índice, a parte de treino recebe a parte inicial
                            e o treino a parte restante, de acordo com o
                            valor em test_size.
    target_col : string
        Nome do target.
    features_cols : list, opcional
        Lista contendo os nomes das features. O valor padrão é None.
    cat_cont_var_bins : integer, opcional
        Quantidade de bins para a divisão do target contí­nuo e assim criar a
        separação treino-teste estratificada. Usada apenas quando
        sep_type="regression". O valor padrão é 5.
    test_size : float, opcional
        Tamanho da amostra de teste, dado em fração do total de registros do
        DataFrame de entrada. O valor padrão é  0.2.
    lower_limit_date_train : string, opcional
        Data limite inferior para o treino, para sep_type='time_series' e 
        test_size=None
    upper_limit_date_train : string, opcional
        Data limite superior para o treino, para sep_type='time_series' e 
        test_size=None
    lower_limit_date_test : string, opcional
        Data limite inferior para o teste, para sep_type='time_series' e 
        test_size=None
    upper_limit_date_test : string, opcional
        Data limite superior para o teste, para sep_type='time_series' e 
        test_size=None
    random_state : integer ou RandomState, opcional
         Gerador numérico para ser usado para a geração da amostra
         aleatória, para sep_type='classification' ou
         sep_type='regression'. O valor padrão é 10.

    Returns
    -------
    X_train : pandas DataFrame
        DataFrame contendo as features de treino.
    y_train : pandas DataFrame
        DataFrame contendo o target de treino.
    X_test : pandas DataFrame
        DataFrame contendo as features de teste.
    y_test : pandas DataFrame
        DataFrame contendo o target de teste.

    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if features_cols is None:
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

        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        if not is_datetime(df_temp.index):
            raise Exception('O índice do DataFrame deve estar no ' +
                            'formato datetime.')

        df_temp.sort_index(inplace=True)
        
        if test_size is not None:

            train_size = int(df_temp.shape[0] * (1.-test_size))
            train = df_temp.iloc[:train_size, :]
            test = df_temp.iloc[train_size:, :]
            
        elif ((test_size is None) and (lower_limit_date_train is not None) and
              (upper_limit_date_train is not None) and 
              (lower_limit_date_test is not None) and
              (upper_limit_date_test is not None)):
            
            train = df_temp.loc[lower_limit_date_train:
                                upper_limit_date_train, :]
            test = df_temp.loc[lower_limit_date_test:
                               upper_limit_date_test, :]
        else:
            raise Exception('Caso o valor de test_size seja None os valores ' +
                            ' limites de datas devem ser especificados.')

    else:
        raise Exception('Valor inválido no parâmetro sep_type ' +
                        '(somente "classification", "regression" e ' +
                        '"time_series" são permitidos)')

    X_train = train[features_cols].copy()
    y_train = train[[target_col]].copy()
    X_test = test[features_cols].copy()
    y_test = test[[target_col]].copy()

    return X_train, y_train, X_test, y_test
