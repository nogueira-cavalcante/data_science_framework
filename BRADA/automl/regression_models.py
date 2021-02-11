"""Modelos de classificação implementados."""


def regression_models(random_state):
    """
    Implementação de modelos de regressão para o automl.

    Esta função contempla os modelos de regressão que serão usados assim
    como a determinação de parâmetros padrões para cada modelo.

    Parameters
    ----------
    random_state : interger ou RandomState
        Gerador numérico para ser usado para a geração da amostra aleatória.
        Esse gerador é o mesmo que será usado no algoritmo de automl.

    Returns
    -------
    implemented_models : dict
        Dicionário de modelos de regressão.
    default_parameter_space : dict
        Dicionário de parâmetros padrões de modelos de classificação.

    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['random_forest'] = RandomForestRegressor
    default_parameter_space['random_forest'] = {'n_estimators': [100],
                                                'random_state': [random_state]}

    implemented_models['extra_tree'] = ExtraTreesRegressor
    default_parameter_space['extra_tree'] = {'n_estimators': [100],
                                             'random_state': [random_state]}

    implemented_models['ada_boost'] = AdaBoostRegressor
    default_parameter_space['ada_boost'] = {'n_estimators': [50],
                                            'random_state': [random_state]}

    implemented_models['gbm'] = GradientBoostingRegressor
    default_parameter_space['gbm'] = {'n_estimators': [100],
                                      'random_state': [random_state]}

    implemented_models['xg_boost'] = XGBRegressor
    default_parameter_space['xg_boost'] = {'n_estimators': [50],
                                           'random_state': [random_state]}

    return implemented_models, default_parameter_space
