def automl(X_train, y_train, training_type, info_model_matrix, n_splits=2,
           validation_size=0.1, random_state=10):

    import time
    import inspect
    import numpy as np
    import pandas as pd
    from skopt import BayesSearchCV
    from sklearn.metrics import make_scorer

    if (validation_size <= 0.0) or (validation_size >= 0.5):
        raise Exception('O valor em validation_size deve ser maior ' +
                        'que 0 e menor que 0.5.')
    else:
        train_size = int(X_train.shape[0] * (1 - validation_size))

    if training_type == 'time_series':

        from .implemented_models import models_time_series
        implemented_models, default_parameter_space = models_time_series()

        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=n_splits,
                             max_train_size=train_size)

        from sklearn.metrics import r2_score
        scorer = make_scorer(r2_score,
                             greater_is_better=True)

        score_column_name = 'best_r2_score'

    elif training_type == 'regression':

        from .implemented_models import models_regression
        implemented_models, default_parameter_space = models_regression()

        from sklearn.model_selection import ShuffleSplit
        cv = ShuffleSplit(n_splits=n_splits,
                          train_size=train_size,
                          random_state=random_state)

        from sklearn.metrics import r2_score
        scorer = make_scorer(r2_score,
                             greater_is_better=True)

        score_column_name = 'best_r2_score'

    elif training_type == 'classification':

        from .implemented_models import models_classification
        implemented_models, default_parameter_space = models_classification()

        from sklearn.model_selection import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(n_splits=n_splits,
                                    train_size=train_size,
                                    random_state=random_state)

        from sklearn.metrics import roc_auc_score
        scorer = 'roc_auc_ovr_weighted'

        score_column_name = 'best_roc_auc_score'

    else:
        raise Exception('O tipo de treinamento especificado em ' +
                        'training_type ( ' + training_type + ') ' +
                        'é inválido. Escolha entre "time_series", ' +
                        '"regression" ou "classification".')

    print('Validação dos parâmetros informados...')

    model_name_list = []
    model_list = []
    space_parameter_list = []
    n_iter_list = []
    for info_model in info_model_matrix:

        # Verificando se o modelo informado está implementado.
        name_model = info_model[0]
        try:
            model_list.append(implemented_models[name_model])
            model_name_list.append(name_model)
        except KeyError:
            raise Exception('O modelo ' + name_model + '  é inválido.' +
                            ' Consulte a relação de modelos implementados ' +
                            'na documentação desta função.')

        # Verificando se os parâmetros passados existem na função.

        if len(list(info_model[1].values())) == 0:
            kwargs_model = default_parameter_space[name_model]
        else:
            kwargs_model = info_model[1]
            argspec = inspect.getfullargspec(implemented_models[name_model])
            if False in np.isin(list(kwargs_model.keys()), argspec.args):
                raise Exception('Um ou mais parâmetros informados para ' +
                                'modelo ' + name_model + ' são inválidos. ' +
                                'Revise os parâmetros esperados para este ' +
                                'modelo específico.')
        space_parameter_list.append(kwargs_model)

        # Verificando a quantidade de iterações.
        n_iter = info_model[2]
        quant = 1
        parameters_values = list(kwargs_model.values())
        for parameters in parameters_values:
            quant = quant * len(parameters)
        if (n_iter > 0) and (n_iter <= quant):
            n_iter_list.append(n_iter)
        else:
            raise Exception('O número de iterações deve ser maior que ' +
                            'zero ou menor ou igual ao total de combinações ' +
                            'do espaço de parâmetros (' + str(quant) + ')')

    print('Validação concluída com sucesso!')
    print('\n')

    # Início do cronômetro:
    start_time = time.time()

    bayes_search_list = []
    bayes_best_score = []
    for i in range(len(model_list)):

        print('Modelo em execução:', model_name_list[i])

        # Instânciando a classe do estimador
        estimator = model_list[i]
        estimator = estimator()

        bayes_search = BayesSearchCV(estimator=estimator,
                                     search_spaces=space_parameter_list[i],
                                     n_jobs=-1,
                                     n_iter=n_iter_list[i],
                                     scoring=scorer,
                                     verbose=0,
                                     cv=cv,
                                     random_state=random_state,
                                     )
        bayes_search.fit(X=X_train, y=y_train)
        bayes_best_score.append(bayes_search.best_score_)
        bayes_search_list.append(bayes_search)

    print('\n')
    tempo_total = (time.time() - start_time)/60.
    print('Tempo total de execução: ' + str(round(tempo_total, 2)) +
          ' minutos.')

    df_best_models = pd.DataFrame(data={'name_models': model_name_list,
                                        score_column_name: bayes_best_score,
                                        'model': bayes_search_list})
    df_best_models.sort_values(by=score_column_name, ascending=False,
                               inplace=True)

    return df_best_models
