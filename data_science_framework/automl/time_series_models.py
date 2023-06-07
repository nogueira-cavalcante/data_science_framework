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
    from .custom_models.holt_winters import HoltWinters
    from .custom_models.sarimax import Sarimax
    from .custom_models.xgb_regressor import Xgb_Regressor
    from .custom_models.neural_prophet import Neural_Prophet
    from .custom_models.nbeats import NBeats_Keras
    from .custom_models.voting_holt_winters_sarimax import Voting_HoltWinters_Sarimax
    from .custom_models.voting_sarimax_xgb_regressor import Voting_Sarimax_XGBRegressor

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['holt_winters'] = HoltWinters
    default_parameter_space['holt_winters'] = {'trend': [None]}

    implemented_models['sarimax'] = Sarimax
    default_parameter_space['sarimax'] = {'p': [1]}

    implemented_models['xgb_regressor'] = Xgb_Regressor
    default_parameter_space['xgb_regressor'] = {'n_estimators': [100]}

    implemented_models['neural_prophet'] = Neural_Prophet
    default_parameter_space['neural_prophet'] = {'growth': ['linear']}

    implemented_models['nbeats'] = NBeats_Keras
    default_parameter_space['nbeats'] = {'verbose': [True]}
    
    implemented_models['voting_holt_winters_sarimax'] = Voting_HoltWinters_Sarimax
    default_parameter_space['voting_holt_winters_sarimax'] = {'sx_weight': [0.5]}

    implemented_models['voting_sarimax_xgb_regressor'] = Voting_Sarimax_XGBRegressor
    default_parameter_space['voting_sarimax_xgb_regressor'] = {'sx_weight': [0.5]}

    return implemented_models, default_parameter_space
