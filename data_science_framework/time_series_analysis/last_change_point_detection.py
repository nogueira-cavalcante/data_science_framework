# -*- coding: utf-8 -*-

def last_change_point_detection(df, model='rbf', custom_cost=None, min_size=2,
                                jump=5, pen=5, min_final_df_size=None):
    """
    Indentificador de quebra de tendência em séries temporais.
    
    Esta função indentifica as quebras de tendência na série temporal informada
    e retorna uma sub-amostra da série original a partir da última
    quebra de tendência identificada. Caso a série resultante a partir da
    última quebra de tendência for menor que o valor informado em
    min_final_df_size então será retornado as últimas min_final_df_size
    observações da série temporal original. Para que essa função funcione
    devidamente a série temporal informada precisa estar ordenada.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo a série temporal.
    model : string, opcional
        Modelo usado na função ruptures para achar os breakpoints. O valor
        padrão é 'rbf'. Não é usado quando 'custom_cost' não é None. Os
        valores possíveis são:
            - 'l1';
            - 'l2';
            - 'rbf'.
    custom_cost : BaseCost, opcional
        Função custom cost. O valor padrão é None.
    min_size : integer, opcional
        Comprimento mínimo do segmento, usado na função ruptures para achar os
        breakpoints. O valor padrão é 2.
    jump  : integer, opcional
        Subamostra (um cada ponto pulado). O valor padrão é 5.
    pen : integer, opcional
        Valor da penalidade, usado na função ruptures para achar os
        breakpoints. O valor padrão é 5.
    min_final_df_size : integer, opcional
        Valor mínimo para a série temporal resultante. Ou seja, caso a
        série temporal resultante da última quebra de tendência tiver um
        tamanho menor que min_final_df_size então a função retornará as 
        últimas min_final_df_size observações da série temporal original.
        O valor padrão é None.

    Returns
    -------
    pandas DataFrame
        Série temporal resultante, ou sendo a última quebra de tendência da 
        série orginal ou sendo as última min_final_len da série original.
        
    """
    import ruptures as rpt
    import pandas as pd
    
    if (type(df) != pd.core.frame.DataFrame) or (df.shape[1] != 1):
        raise Exception('A série temporal informada precisa estar no formato'
                        + ' DataFrame, possuindo apenas uma coluna.')

    cpd = rpt.Pelt(model=model, custom_cost=custom_cost, min_size=min_size,
                   jump=jump)
    cpd.fit(df)
    change_points = cpd.predict(pen=pen)

    if len(change_points) > 1:
        # A última observação tem sido sempre o last point change. Por isso
        # consideramos o penúltimo point change.
        change_point = change_points[-2]
        df_temp = df.iloc[change_point:].copy()

    else:
        df_temp = df.copy()

    if min_final_df_size:
        if type(min_final_df_size) is int:
            if min_final_df_size > len(df_temp):
                df_temp = df.iloc[-min_final_df_size:].copy()

        else:
            raise Exception('O valor em min_final_len quando especificado' + 
                            'deve ser um número inteiro!')

    return df_temp