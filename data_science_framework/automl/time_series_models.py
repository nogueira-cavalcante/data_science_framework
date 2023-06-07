"""Modelos implementados de séries temporais."""


def time_series_models(random_state):
    """
    Implementação de modelos de séries temporais para o automl.

    Esta função contempla os modelos de séries temporais que serão usados assim
    como a determinação de parâmetros padrões para cada modelo.

    Parameters
    ----------
    random_state : interger ou RandomState
        Gerador numérico para ser usado para a geração da amostra aleatória.
        Esse gerador é o mesmo que será usado no algoritmo de automl.

    Returns
    -------
    implemented_models : dict
        Dicionário de modelos de séries temporais.
    default_parameter_space : dict
        Dicionário de parâmetros padrões de modelos de séries temporais.

    """
    from .custom_models.sarimax import Sarimax

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['sarimax'] = Sarimax
    default_parameter_space['sarimax'] = {'p': [1]}

    return implemented_models, default_parameter_space
