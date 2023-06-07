# -*- coding: utf-8 -*-
"""Seleção de variáveis utilizando o teste de hipótese de Granger."""


def granger_causality(df, target_col, lags=12, max_p_value=0.05,
                      shift_feature=True, verbose=True):
    """
    Seleção de variáveis utilizando o teste de hipótese de Granger.

    Esta função retorna um DataFrame filtrando as variáveis que passam no
    teste de hipóteses da causalidade de Granger.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo o target e as features.
    target_col : string
        Nome da coluna target.
    lags : int, optional
        Valor total de lags a ser testado. O padrão é 12.
    max_p_value : float, optional
        Valor do p-valor máximo para que o teste desconsidere a hipótese nula
        e passe a considerar a hipótese alternativa (há causalidade).
        O valor padrão é 0.05.
    shift_feature : boolean, optional
        Se True a função criará uma nova variável deslocada em lags
        correspondente ao melhor lag do teste de Granger (o de menor p-valor).
        O valor padrão é True.
    verbose : boolean, optional
        Se True mostra na tela os resultados. O valor padrão é True.

    Returns
    -------
    pandas DataFrame
        DataFrame contendo o target com as features selecionadas.

    """
    import numpy as np
    from statsmodels.tsa.stattools import grangercausalitytests
    import warnings

    df_temp = df.copy()
    
    feature_cols = df_temp.drop(columns=target_col).columns

    for feature_col in feature_cols:

        lag_list = []

        warnings.filterwarnings("ignore")
        gc_res = grangercausalitytests(df_temp[[target_col,
                                                feature_col]],
                                       lags, verbose=False)

        for lag in np.arange(1, lags+1):

            p_value_from_f_test = gc_res[lag][0]['ssr_ftest'][1]
            lag_list.append(p_value_from_f_test)

        best_lags = np.array(lag_list)[np.array(lag_list) <= max_p_value]

        if (len(best_lags) > 0) and (shift_feature is True):

            if verbose:
                print('Good Granger feature (shifted):', feature_col)

            lag_to_shift = lag_list.index(min(lag_list)) + 1
            new_feature_name = feature_col + '_lag' + str(lag_to_shift)
            df_temp_shifted = df_temp.shift(lag_to_shift).copy()
            df_temp.loc[:, new_feature_name] = df_temp_shifted[feature_col]
            df_temp.drop(feature_col, axis=1, inplace=True)

        elif (len(best_lags) > 0) and (shift_feature is False):

            if verbose:
                print('Good Granger feature (non-shifted):', feature_col)

        else:
            if verbose:
                print('Bad Granger feature:', feature_col)
            df_temp.drop(columns=feature_col, inplace=True)

    df_temp.dropna(inplace=True)

    return df_temp
