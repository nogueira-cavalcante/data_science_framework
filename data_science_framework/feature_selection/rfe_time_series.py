# -*- coding: utf-8 -*-
"""Seleção de variáveis utilizando o Recursive Feature Elimination."""


def rfe_time_series(df, target_col, df_start_end_lags=None,
                    start_lag=1, end_lag=12,
                    method_find_best_lag='pearson',
                    drop_original=True,
                    n_features_to_select=4, verbose=False,
                    random_state=10):
    """
    Seleção de features para séries temporais.

    Esta função utiliza o Recursive Feature Elimination (RFE) juntamento com o
    estimador de regressão Random Forest para estimar quais são as lags dos
    melhores variáveis que ajudam a prever o variável alvo. A escolha do
    melhor lag é feita ou através de correlação pearson ou através do RFE
    (dependendo do parâmetro de entrada method_find_best_lag). A escolha da
    melhor variável é feita através do RFE.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame de entrada.
    target_col : string
        Nome da coluna target.
    df_start_end_lags : pandas DataFrame, optional
        DataFrame contendo as features com as informações iniciais e finais
        de lags que serão investigados. IMPORTANTE: essas informações devem
        ser valores inteiros. Esta informação é útil quando o intervalo de
        busca de lags das features for personalizado para cada features.
        O índices devem ser os nomes das features, uma coluna chamada de
        'start_lag' e uma outra coluna chamada 'end_lag'. O valor padrão é
        None. Exemplo de criação:
            - df_start_end_lags = pd.DataFrame()
            - df_start_end_lags.loc['A', ['start_lag', 'end_lag']] = 0, 4
            - df_start_end_lags.loc['B', ['start_lag', 'end_lag']] = 1, 5
            - df_start_end_lags.loc['C', ['start_lag', 'end_lag']] = 2, 6
    start_lag : int, opcional
        Lag inicial para busca da melhor lag. Caso df_start_end_lags for None
        esse parâmetro será usado igualmente para todas as features.
        O valor padrão é 1.
    end_lag : int, opcional
        Lag final para busca da melhor lag. Caso df_start_end_lags for None
        esse parâmetro será usado igualmente para todas as features.
        O valor padrão é 12.
    method_find_best_lag : string, opcional
        Método de busca da melhor lag por feature. Atualmente suporta dois
        métodos:
            - 'pearson';
            - 'rfe'.
        O valor padrão é 'pearson'.
    drop_original : boolean, opcional
        Caso True a variável feature original será removida. O valor padrão é
        True.
    n_features_to_select : int, opcional
        Quantidade máxima de features finais selecionadas.O padrão pe 4.
    verbose : boolean, opcional
        Caso True imprime na tela a melhor defasagem de cada feature assim
        e a rodada do RFE. O valor padrão é False.
    random_state : int ou RandomState, opcional
        Gerador numérico para ser usado para a geração da amostra aleatória.
        O valor padrão é 10.

    Returns
    -------
    pandas DataFrame
        DataFrame contendo a varíável target juntamente com as melhores
        features selecionadas das melhores lags.

    """
    import warnings
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor

    df_temp = df.copy()

    if df_temp.isnull().sum().sum() > 1:
        raise Exception('O DataFrame de entrada não deve conter nulos.')

    reg = RandomForestRegressor(n_estimators=10, random_state=random_state)

    feature_columns = df_temp.drop(target_col, axis=1).columns

    for feature in feature_columns:

        if df_start_end_lags is not None:

            start_lag = df_start_end_lags.loc[feature, 'start_lag']
            end_lag = df_start_end_lags.loc[feature, 'end_lag']
            
            start_lag = start_lag.astype('int64')
            end_lag = end_lag.astype('int64')

        lags_range = np.arange(start_lag, end_lag + 1)

        if method_find_best_lag == 'pearson':

            shifts_list = []
            coefs_list = []

            for lag in lags_range:

                shifts_list.append(lag)

                coef = pearsonr(df_temp.loc[df_temp.index[lag]:, target_col],
                                df_temp[feature].shift(lag).dropna())[0]
                coefs_list.append(np.abs(coef))

            best_coef_index = coefs_list.index(max(coefs_list))
            best_lag = shifts_list[best_coef_index]

        elif method_find_best_lag == 'rfe':

            df_lags = df_temp[[target_col]].copy()

            for lag in lags_range:

                lagged_feature = df_temp[feature].shift(lag)

                df_lags.loc[:, feature + '_lag' + str(lag)] = lagged_feature

            df_lags.dropna(inplace=True)

            X_lag = df_lags.drop(target_col, axis=1).copy()
            y_lag = df_lags[[target_col]].copy()

            warnings.filterwarnings("ignore")
            rfe_best_lag = RFE(reg, n_features_to_select=1, verbose=False)
            fitted_rfe_best_lag = rfe_best_lag.fit(X=X_lag, y=y_lag)

            best_lag = lags_range[fitted_rfe_best_lag.ranking_ == 1][0]

        new_feature_name = feature + '_lag' + str(best_lag)
        df_temp.loc[:, new_feature_name] = df_temp[feature].shift(best_lag)

        if drop_original:
            df_temp.drop(feature, axis=1, inplace=True)

        if verbose:
            print('Melhor defasagem para ' + feature + ':', best_lag)

    if verbose:
        print('\n')

    df_temp.dropna(inplace=True)

    X_features = df_temp.drop(target_col, axis=1).copy()
    y_features = df_temp[[target_col]].copy()

    warnings.filterwarnings("ignore")
    rfe_features = RFE(reg,
                       n_features_to_select=n_features_to_select,
                       verbose=verbose)
    fitted_rfe_features = rfe_features.fit(X_features, y_features)

    cond_ranking = fitted_rfe_features.ranking_ == 1
    df_temp = X_features[X_features.columns[cond_ranking]].copy()
    df_temp.loc[:, target_col] = y_features[target_col].copy()

    return df_temp
