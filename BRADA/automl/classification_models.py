"""Modelos de classificação implementados."""


def classification_models(random_state):
    """
    Implementação de modelos de classificação para o automl.

    Esta função contempla os modelos de classificaçção que serão usados assim
    como a determinação de parâmetros padrões para cada modelo.

    Parameters
    ----------
    random_state : interger ou RandomState
        Gerador numérico para ser usado para a geração da amostra aleatória.
        Esse gerador é o mesmo que será usado no algoritmo de automl.

    Returns
    -------
    implemented_models : dict
        Dicionário de modelos de classificação.
    default_parameter_space : dict
        Dicionário de parâmetros padrões de modelos de classificação.

    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['logistic_regression'] = LogisticRegression
    default_parameter_space['logistic_regression'] = {'penalty': ['l2'],
                                                      'random_state':
                                                          [random_state]}

    implemented_models['svc'] = SVC
    default_parameter_space['svc'] = {'C': [1], 'probability': [True],
                                      'random_state': [random_state]}

    implemented_models['random_forest'] = RandomForestClassifier
    default_parameter_space['random_forest'] = {'n_estimators': [100],
                                                'random_state': [random_state]}

    return implemented_models, default_parameter_space
