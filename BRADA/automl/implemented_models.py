def models_time_series():

    from .custom_models.sarimax import Sarimax

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['sarimax'] = Sarimax
    default_parameter_space['sarimax'] = {'p': [1]}

    return implemented_models, default_parameter_space


def models_regression():

    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['random_forest'] = RandomForestRegressor
    default_parameter_space['random_forest'] = {'n_estimators': [100]}

    implemented_models['extra_tree'] = ExtraTreesRegressor
    default_parameter_space['extra_tree'] = {'n_estimators': [100]}

    implemented_models['ada_boost'] = AdaBoostRegressor
    default_parameter_space['ada_boost'] = {'n_estimators': [50]}

    implemented_models['gbm'] = GradientBoostingRegressor
    default_parameter_space['gbm'] = {'n_estimators': [100]}

    implemented_models['xg_boost'] = XGBRegressor
    default_parameter_space['xg_boost'] = {'n_estimators': [50]}

    return implemented_models, default_parameter_space


def models_classification():

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    implemented_models = dict()
    default_parameter_space = dict()

    implemented_models['logistic_regression'] = LogisticRegression
    default_parameter_space['logistic_regression'] = {'penalty': ['l2']}

    implemented_models['svc'] = SVC
    default_parameter_space['svc'] = {'C': [1], 'probability': [True]}

    implemented_models['random_forest'] = RandomForestClassifier
    default_parameter_space['random_forest'] = {'n_estimators': [100]}

    return implemented_models, default_parameter_space
