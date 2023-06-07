"""Otimizador de hiperparâmetros automático."""


def optimized_search(X_train, y_train, X_test, y_test, training_type,
                     info_model_matrix, random_state=10):
    """
    Otimizador de hiperparâmetros automático.

    Esta função utiliza o pacote skopt para a otimização dos hiperparâmetros,
    tentando primeiramente o otimizador bayesiano com processos gaussianos.
    Caso falhe tentará a otimização baseada em árvores GRB.
    Atualmente, para cada tipo de treinamento (training_type) esta função
    suporta os seguintes modelos:
        - time_series:
            - 'holt_winters';
            - 'sarimax';
            - 'xgb_regressor';
            - 'neural_prophet';
            - 'nbeats'.
        - classification:
            - 'logistic_regression';
            - 'svc';
            - 'random_forest'.
        - regression:
            - 'ada_boost';
            - 'extra_tree';
            - 'gbm';
            - 'random_forest';
            - 'xg_boost'.

    Parameters
    ----------
    X_train : pandas DataFrame
        DataFrame contendo as features de treinamento.
    y_train : pandas DataFrame
        DataFrame contendo o target de treinamento.
    X_test : pandas DataFrame
        DataFrame contendo as features de teste.
    y_test : pandas DataFrame
        DataFrame contendo o target de teste.
    training_type : string
        Tipo de treinamento ('time_series', 'classification' ou 'regression').
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
    random_state : integer ou RandomState, opcional
        Gerador numérico para ser usado para a geração da amostra aleatória.
        O valor padrão é 10.

    Raises
    ------
    Exception
        - Tipode de otimizador em optimization_kind diferente de 'gaussian'
          ou 'forest'
        - Tipo de treinamento em training_type for diferente de
          "classification", "regression" ou "time_series";
        - Nome do modelo não corresponder aos implementados;
        - Parâmetros de um determinado modelo não existir nos modelos
          implementados;
        - Números de iterações for inferior a 10.

    Returns
    -------
    pandas DataFrame
        DataFrame de resultado contendo como índice os nomes dos modelos
        e com as seguintes coluynas:
            - best_fitted_model: melhor modelo considerando apenas os valores
                                 de treino;
            - best_refitted_model: melhor modelo considerando apenas os
                                   valores de treino e teste;

            - best_parameters: dicionário contendo os melhores hiperparâmetros
                               dos modelos;
            - train_score: valor da melhor métrica, por modelo, pora os valores
                           de treinamento (roc_aux para 'classification' e
                           mse para 'regression' e 'time_series');
            - test_score: valor da melhor métrica, por modelo, pora os valores
                           de teste (roc_aux para 'classification' e
                           mse para 'regression' e 'time_series').
    """
    import time
    import inspect
    import warnings
    import numpy as np
    import pandas as pd
    from skopt.utils import use_named_args
    from skopt import gp_minimize, gbrt_minimize

    # Usado para o refit
    X_complete = pd.concat([X_train, X_test]).copy()
    y_complete = pd.concat([y_train, y_test]).copy()

    if training_type == 'time_series':

        from .time_series_models import time_series_models
        im, dps = time_series_models(random_state)
        implemented_models, default_parameter_space = im, dps

        from sklearn.metrics import mean_squared_error
        scorer = mean_squared_error
        score_column_name = 'mse_score'

    elif training_type == 'classification':
        raise Exception('Ainda não implementado!')

    elif training_type == 'regression':
        raise Exception('Ainda não implementado!')

    else:
        raise Exception('O tipo de treinamento especificado em ' +
                        'training_type ( ' + training_type + ') ' +
                        'é inválido. Escolha entre "time_series", ' +
                        '"regression" ou "classification".')

    model_names_list = []
    models_list = []
    space_parameters_list = []
    n_iters_list = []

    for info_model in info_model_matrix:

        # Verificando se o modelo informado está implementado.
        name_model = info_model[0]
        try:
            models_list.append(implemented_models[name_model])
            model_names_list.append(name_model)
        except KeyError:
            raise Exception('O modelo ' + name_model + '  é inválido.' +
                            ' Consulte a relação de modelos implementados ' +
                            'na documentação desta função.')

        # Verificando se os parâmetros passados existem na função.
        if (len(info_model[1]) == 0) or info_model[1] is None:
            kwargs_model = default_parameter_space[name_model]
        else:
            kwargs_model = info_model[1]
            parameter_names_list = []
            for parameter in kwargs_model:
                parameter_names_list.append(parameter.name)
            argspec = inspect.getfullargspec(implemented_models[name_model])
            if False in np.isin(parameter_names_list, argspec.args):
                raise Exception('Um ou mais parâmetros informados para ' +
                                'modelo ' + name_model + ' são inválidos. ' +
                                'Revise os parâmetros esperados para este ' +
                                'modelo específico.')
        space_parameters_list.append(kwargs_model)

        # Obtendo a quantidade de iterações por modelo.
        n_iter = info_model[2]
        n_iters_list.append(n_iter)

    print('Validação concluída com sucesso!')
    print('\n')

    best_model_fitted_list = []
    best_model_refitted_list = []

    score_train_list = []
    score_test_list = []

    best_parameters = []

    for i in range(len(models_list)):

        print('Modelo em execução:', model_names_list[i])

        # Início do cronômetro:
        start_time = time.time()

        # Instânciando a classe do estimador
        estimator = models_list[i]

        # Listas com os modelos fitted e refitted
        model_fitted_list = []
        model_refitted_list = []

        space = space_parameters_list[i]

        @use_named_args(space)
        def objective(**params):

            warnings.filterwarnings("ignore")

            estimator_fit = estimator()
            estimator_fit.set_params(**params)
            estimator_fit.fit(X_train, y_train)

            model_fitted_list.append(estimator_fit)

            score = scorer(y_test, estimator_fit.predict(X_test))

            estimator_refit = estimator()
            estimator_refit.set_params(**params)
            estimator_refit.fit(X_complete, y_complete)

            model_refitted_list.append(estimator_refit)

            return score

        n_calls = n_iters_list[i]

        try:
            optimized = gp_minimize(objective,
                                    space,
                                    n_calls=n_calls,
                                    n_jobs=-1,
                                    random_state=random_state)
            optimization_kind = 'Otimização bayesiana concluída.'

        except:
            optimized = gbrt_minimize(objective,
                                      space,
                                      n_calls=n_calls,
                                      n_jobs=-1,
                                      random_state=random_state)
            optimization_kind = 'Otimização sequencial (árvores GRB) concluída'

        print(optimization_kind)

        boolean_list = list(optimized.func_vals == optimized.fun)
        index_best_model = boolean_list.index(True)

        best_model_fit = model_fitted_list[index_best_model]
        best_model_refit = model_refitted_list[index_best_model]

        best_model_fitted_list.append(best_model_fit)
        best_model_refitted_list.append(best_model_refit)

        score_train = scorer(y_train, best_model_fit.predict(X_train))
        score_test = scorer(y_test, best_model_fit.predict(X_test))

        score_train_list.append(score_train)
        score_test_list.append(score_test)

        best_params = dict()
        for i in range(len(space)):
            best_params[space[i].name] = optimized.x[i]
        best_parameters.append(best_params)

        tempo_total = (time.time() - start_time)/60.
        print('Tempo total de execução: ' + str(round(tempo_total, 2)) +
              ' minutos.')
        print('\n')

    score_column_name_train = 'train_' + score_column_name
    score_column_name_test = 'test_' + score_column_name

    df_best_models = pd.DataFrame(data=np.array([best_model_fitted_list,
                                                 best_model_refitted_list,
                                                 best_parameters,
                                                 score_train_list,
                                                 score_test_list]).transpose(),
                                  columns=['best_fitted_model',
                                           'best_refitted_model',
                                           'best_parameters',
                                           score_column_name_train,
                                           score_column_name_test],
                                  index=model_names_list)

    df_best_models.sort_values(by=score_column_name_test,
                               ascending=True,
                               inplace=True)

    return df_best_models
