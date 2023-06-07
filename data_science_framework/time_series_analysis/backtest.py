# -*- coding: utf-8 -*-
"""Backtest para séries temporais."""


def backtest(df, target_col, info_model_matrix, backtest_window=6,
             forecast_step=1, forecast_window=1, freq='MS', alpha=0.05):
    """
    Backtest para séries temporais.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo as features e o target. A informação de tempo 
        deve ser fornecida como índice.
    target_col : string
        Nome da variável target.
    info_model_matrix : two-dimensional list (array)
        Matriz do tipo lista(ou do tipo numpy array). Cada linha da matriz
        deve conter:
            - string: nome do modelo desejado implementado;
            - list ou numpy array: espaço de parâmetros para o modelo
              escolhido;
            - integer: quantidade de iterações para o otimizador.
        Exemplo:

            from skopt.space import Real, Integer, Categorical

            hw_param_space = [Categorical([None, 'add', 'mul'], name='trend'),
                              Categorical(['add', 'mul'], name='seasonal')]

            sarimax_param_space = [Categorical([1, 2, 3], name='p'),
                                   Real([0, 1], name='d'),
                                   Integer([1, 2, 3], name='q')]

            info_model_matrix=[
                               ['holt_winters', hw_param_space, 10],
                               ['sarimax'; sarimax_param_space, 15]
                              ]
    backtest_window : integer, opcional
        Valor da janela do backtest. O valor padrão é 6.
    forecast_step : integer, opcional
        Quantidade de passos futuros para a previsão. O valor padrão é 1.
    forecast_window : interger, opcional
        Dado o valor de forecast_step o valor em forecast_window representa 
        a quantidade que será usada pela função de forecast_step. Por exemplo,
        caso forecast_step seja 6 a função estimará 6 passos de previsões
        futuras. Caso forecast_window seja 2 então desses 6 passos serão
        considerados as 5ª e 6ª previsões.
    freq : string, opcional
        Tipo de frequência do índice data. Atualmente essa função só comporta
        a frequência 'MS'.
    alpha : TYPE, opcional
        Os intervalos de confiança das previsões serão (1 - alpha), caso 
        o método escolhido permita. Caso contrário, os intervalos de confiança
        retornados serão 0. O valor padrão é 0.05.

    Returns
    -------
    pandas DataFrame
        Resultado do backtest, contendo as seguintes colunas:
            - reference_time: tempo de referência da previsão;
            - best_model: melhor modelo escolhido;
            - pred_time: tempo da previsão;
            - true_y_validation: valor verdadeiro de validação;
            - true_y_test: valor verdadeiro de teste;
            - pred_y_validation: valor previsto de validação;
            - pred_y_validation_lower_limit: limite inferior da previsão de validação;
            - pred_y_validation_upper_limit: limite superior da previsão de validação;
            - pred_y_test: valor previsto do teste;
            - pred_y_test_lower_limit: limite inferior da previsão de teste;
            - pred_y_test_upper_limit: limite superior da previsão de teste.

    """
    import numpy as np
    import pandas as pd
    from pandas.tseries.offsets import DateOffset
    from ..train_test_split.generic_train_test_split import generic_train_test_split
    from ..automl.optimized_search import optimized_search
        
    if forecast_step > backtest_window:
        raise Exception('O valor em forecast_step deve ser menor ou igual ao valor de backtest_window!')
    
    if forecast_window > forecast_step:
        raise Exception('O valor em forecast_window deve ser menor ou igual ao valor de forecast_step!')
        
    df_temp = df.copy()
    
    try:
        df_temp.index = pd.DatetimeIndex(df_temp.index, freq=freq)
        df_temp.sort_index(inplace=True)
    
    except:
        raise Exception('O índice deve estar no formato data!')
    
    min_date = df_temp.index.min()
    max_date = df_temp.index.max()
        
    if freq == 'MS':
        
        lower_limit_date_train_list = min_date
        upper_limit_date_train_list = pd.date_range(start=(max_date - DateOffset(months=forecast_step-1)) - DateOffset(months=backtest_window),
                                                    end=(max_date - DateOffset(months=forecast_step)),
                                                    freq=freq)
        lower_limit_date_test_list = upper_limit_date_train_list + DateOffset(months=1)
        upper_limit_date_test_list = upper_limit_date_train_list + DateOffset(months=forecast_step)
        
    else:
        raise Exception('A frequência especificada (' + freq + ') não existe ou não se encontra implementada!')
    
    reference_list = []
    best_model_list = []
    pred_instant_list = []
    true_y_validation_list = []
    true_y_test_list = []
    pred_y_validation_list = []
    pred_y_validation_lower_limit_list = []
    pred_y_validation_upper_limit_list = []
    pred_y_test_list = []
    pred_y_test_lower_limit_list = []
    pred_y_test_upper_limit_list = []
    
    total = len(upper_limit_date_train_list)
    cont = 1

    for backtest_index in range(len(upper_limit_date_train_list)):
        
        print('########################################')
        print('Backtest de séries temporais:', str(cont) + ' de ' + str(total))
        print('\n')

        (X_train,
         y_train,
         X_test,
         y_test) = generic_train_test_split(df=df_temp,
                                            sep_type='time_series',
                                            target_col=target_col,
                                            features_cols=None,
                                            test_size=None,
                                            lower_limit_date_train=lower_limit_date_train_list,
                                            upper_limit_date_train=upper_limit_date_train_list[backtest_index],
                                            lower_limit_date_test=lower_limit_date_test_list[backtest_index],
                                            upper_limit_date_test=upper_limit_date_test_list[backtest_index]
                                            )
        X_validation = X_train.iloc[-forecast_step:, :].copy()
        y_validation = y_train.iloc[-forecast_step:, :].copy()

        X_train = X_train.iloc[:-forecast_step, :]
        y_train = y_train.iloc[:-forecast_step, :]      
        
        df_best_models = optimized_search(X_train=X_train,
                                          y_train=y_train,
                                          X_test=X_validation,
                                          y_test=y_validation,
                                          training_type='time_series',
                                          info_model_matrix=info_model_matrix,
                                          random_state=10)
 
        # O melhor modelo é o primeiro que aparece no DataFrame df_best_models
        best_time_series_model = df_best_models.index[0]
        
        reference_list = reference_list + list(np.repeat(upper_limit_date_train_list[backtest_index], forecast_window))
        best_model_list = best_model_list + list(np.repeat(best_time_series_model, forecast_window))
        pred_instant_list = pred_instant_list + list(y_test.index.astype(str).to_numpy()[forecast_step-forecast_window:])
        true_y_validation_list = true_y_validation_list + list(y_validation.loc[:, target_col].values[forecast_step-forecast_window:])
        true_y_test_list = true_y_test_list + list(y_test.loc[:, target_col].values[forecast_step-forecast_window:])
        
        try:
            pred_y_validation = df_best_models.loc[best_time_series_model, 'best_fitted_model'].predict(X_validation, return_conf_int=True, alpha=alpha)
            pred_y_test = df_best_models.loc[best_time_series_model, 'best_refitted_model'].predict(X_test, return_conf_int=True, alpha=alpha)
            
            pred_y_validation_list = pred_y_validation_list + list(pred_y_validation.loc[:, target_col].values[forecast_step-forecast_window:])
            pred_y_validation_lower_limit_list = pred_y_validation_lower_limit_list + list(pred_y_validation.loc[:, 'lower_limit'].values[forecast_step-forecast_window:])
            pred_y_validation_upper_limit_list = pred_y_validation_upper_limit_list + list(pred_y_validation.loc[:, 'upper_limit'].values[forecast_step-forecast_window:])
            
            pred_y_test_list = pred_y_test_list + list(pred_y_test.loc[:, target_col].values[forecast_step-forecast_window:])
            pred_y_test_lower_limit_list = pred_y_test_lower_limit_list + list(pred_y_test.loc[:, 'lower_limit'].values[forecast_step-forecast_window:])
            pred_y_test_upper_limit_list = pred_y_test_upper_limit_list + list(pred_y_test.loc[:, 'upper_limit'].values[forecast_step-forecast_window:])
            
        except:
            pred_y_validation = df_best_models.loc[best_time_series_model, 'best_fitted_model'].predict(X_validation)
            pred_y_test = df_best_models.loc[best_time_series_model, 'best_refitted_model'].predict(X_test)
            
            pred_y_validation_list = pred_y_validation_list + list(pred_y_validation.loc[:, target_col].values[forecast_step-forecast_window:])
            pred_y_validation_lower_limit_list = pred_y_validation_lower_limit_list + [0.]
            pred_y_validation_upper_limit_list = pred_y_validation_upper_limit_list + [0.]
            
            pred_y_test_list = pred_y_test_list + list(pred_y_test.loc[:, target_col].values[forecast_step-forecast_window:])
            pred_y_test_lower_limit_list = pred_y_test_lower_limit_list + [0.]
            pred_y_test_upper_limit_list = pred_y_test_upper_limit_list + [0.]

        cont += 1
        
    data = np.array([reference_list,
                     best_model_list,
                     pred_instant_list,
                     true_y_validation_list,
                     true_y_test_list,
                     pred_y_validation_list,
                     pred_y_validation_lower_limit_list,
                     pred_y_validation_upper_limit_list,
                     pred_y_test_list,
                     pred_y_test_lower_limit_list,
                     pred_y_test_upper_limit_list
                    ]).transpose()
    
    columns = ['reference_time',
               'best_model',
               'pred_time',
               'true_y_validation',
               'true_y_test',
               'pred_y_validation',
               'pred_y_validation_lower_limit',
               'pred_y_validation_upper_limit',
               'pred_y_test',
               'pred_y_test_lower_limit',
               'pred_y_test_upper_limit'
              ]
    
    df_backtest = pd.DataFrame(data=data, columns=columns)
    
    df_backtest['mape_validation'] = df_backtest.apply(lambda x: 100 * np.abs(x['true_y_validation'] - x['pred_y_validation']) / x['true_y_validation'], axis=1)
    df_backtest['mape_test'] = df_backtest.apply(lambda x: 100 * np.abs(x['true_y_test'] - x['pred_y_test']) / x['true_y_test'], axis=1)
    
    df_backtest['reference_time'] = pd.to_datetime(df_backtest['reference_time'])
    df_backtest['best_model'] = df_backtest['best_model'].astype(str)
    df_backtest['pred_time'] = pd.to_datetime(df_backtest['pred_time'])
    df_backtest['true_y_validation'] = df_backtest['true_y_validation'].astype(float)
    df_backtest['true_y_test'] = df_backtest['true_y_test'].astype(float)
    df_backtest['pred_y_validation'] = df_backtest['pred_y_validation'].astype(float)
    df_backtest['pred_y_validation_lower_limit'] = df_backtest['pred_y_validation_lower_limit'].astype(float)
    df_backtest['pred_y_validation_upper_limit'] = df_backtest['pred_y_validation_upper_limit'].astype(float)
    df_backtest['pred_y_test'] = df_backtest['pred_y_test'].astype(float)
    df_backtest['pred_y_test_lower_limit'] = df_backtest['pred_y_test_lower_limit'].astype(float)
    df_backtest['pred_y_test_upper_limit'] = df_backtest['pred_y_test_upper_limit'].astype(float)

    print('Backtest realizado com sucesso!')
    
    return df_backtest