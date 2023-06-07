# -*- coding: utf-8 -*-

"""Classe voting unificando os métodos Sarimax e XGBoost."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from .sarimax import Sarimax
from .xgb_regressor import Xgb_Regressor

class Voting_Sarimax_XGBRegressor(BaseEstimator, RegressorMixin):
    """Classe ensemble unificando os métodos Sarimax e Holt-Winters."""
    
    def __init__(self, sx_start_p=2, sx_d=None, sx_start_q=2, sx_max_p=5,
                 sx_max_d=2, sx_max_q=5, sx_start_P=1, sx_D=None, sx_start_Q=1,
                 sx_max_P=2, sx_max_D=1, sx_max_Q=2, sx_max_order=5, sx_m=1,
                 sx_seasonal=True, sx_stationary=False,
                 sx_information_criterion='aic', sx_alpha=0.05, sx_test='kpss',
                 sx_seasonal_test='ocsb', sx_stepwise=True, sx_n_jobs=1,
                 sx_start_params=None, sx_trend=None, sx_method='lbfgs',
                 sx_maxiter=50, sx_offset_test_args=None,
                 sx_seasonal_test_args=None, sx_suppress_warnings=True,
                 sx_error_action='trace', sx_trace=False, sx_random=False,
                 sx_random_state=None, sx_n_fits=10,
                 sx_return_valid_fits=False, sx_out_of_sample_size=0,
                 sx_scoring='mse', sx_scoring_args=None, 
                 sx_with_intercept='auto', sx_sarimax_kwargs=None,
                 sx_weight=0.5, xgb_n_estimators=100, xgb_max_depth=None,
                 xgb_learning_rate=None, xgb_verbosity=None,
                 xgb_objective='reg:squarederror', xgb_booster=None, 
                 xgb_tree_method=None, xgb_n_jobs=None, xgb_gamma=None,
                 xgb_eta=0.3, xgb_min_child_weight=None,
                 xgb_max_delta_step=None, xgb_subsample=None,
                 xgb_colsample_bytree=None, xgb_colsample_bylevel=None,
                 xgb_colsample_bynode=None, xgb_reg_alpha=None,
                 xgb_reg_lambda=None, xgb_scale_pos_weight=None,
                 xgb_base_score=None, xgb_random_state=None,
                 xgb_missing=np.nan, xgb_num_parallel_tree=None,
                 xgb_monotone_constraints=None,
                 xgb_interaction_constraints=None, xgb_importance_type='gain',
                 xgb_gpu_id=None, xgb_validate_parameters=None,
                 xgb_weight=0.5):
        """Inicialização da classe."""
        
        self.Sarimax = Sarimax(start_p=sx_start_p, d=sx_d, start_q=sx_start_q, 
                               max_p=sx_max_p, max_d=sx_max_d, max_q=sx_max_q,
                               start_P=sx_start_P, D=sx_D, start_Q=sx_start_Q,
                               max_P=sx_max_P, max_D=sx_max_D, max_Q=sx_max_Q,
                               max_order=sx_max_order, m=sx_m,
                               seasonal=sx_seasonal, stationary=sx_stationary,
                               information_criterion=sx_information_criterion,
                               alpha=sx_alpha, test=sx_test,
                               seasonal_test=sx_seasonal_test,
                               stepwise=sx_stepwise, n_jobs=sx_n_jobs,
                               start_params=sx_start_params, trend=sx_trend,
                               method=sx_method, maxiter=sx_maxiter,
                               offset_test_args=sx_offset_test_args,
                               seasonal_test_args=sx_seasonal_test_args,
                               suppress_warnings=sx_suppress_warnings,
                               error_action=sx_error_action, trace=sx_trace,
                               random=sx_random, random_state=sx_random_state,
                               n_fits=sx_n_fits,
                               return_valid_fits=sx_return_valid_fits,
                               out_of_sample_size=sx_out_of_sample_size,
                               scoring=sx_scoring, scoring_args=sx_scoring_args,
                               with_intercept=sx_with_intercept,
                               sarimax_kwargs=sx_sarimax_kwargs)
        
        self.Xgb_Regressor = Xgb_Regressor(n_estimators=xgb_n_estimators,
                                           max_depth=xgb_max_depth,
                                           learning_rate=xgb_learning_rate,
                                           verbosity=xgb_verbosity,
                                           objective=xgb_objective,
                                           booster=xgb_booster,
                                           tree_method=xgb_tree_method,
                                           n_jobs=xgb_n_jobs, gamma=xgb_gamma,
                                           eta=xgb_eta, 
                                           min_child_weight=xgb_min_child_weight,
                                           max_delta_step=xgb_max_delta_step,
                                           subsample=xgb_subsample,
                                           colsample_bytree=xgb_colsample_bytree,
                                           colsample_bylevel=xgb_colsample_bylevel,
                                           colsample_bynode=xgb_colsample_bynode,
                                           reg_alpha=xgb_reg_alpha,
                                           reg_lambda=xgb_reg_lambda,
                                           scale_pos_weight=xgb_scale_pos_weight,
                                           base_score=xgb_base_score,
                                           random_state=xgb_random_state,
                                           missing=xgb_missing,
                                           num_parallel_tree=xgb_num_parallel_tree,
                                           monotone_constraints=xgb_monotone_constraints,
                                           interaction_constraints=xgb_interaction_constraints,
                                           importance_type=xgb_importance_type,
                                           gpu_id=xgb_gpu_id,
                                           validate_parameters=xgb_validate_parameters)
        
        
        self.sx_weight = sx_weight
        self.xgb_weight = xgb_weight
        
    def fit(self, X, y):
        """
        Construção do modelo voting para um conjunto de treino (X, y).

        Os índices devem estar no formato datetime. Por exemplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Parameters
        ----------
        X : (pandas DataFrame)
            DataFrame contendo as exógenas.
        y : (pandas DataFrame)
            DataFrame contendo a endógena.
        """
        self.y_train = y.copy()
                                       
        self.Xgb_Regressor.fit(X=X, y=y)
        self.Sarimax.fit(X=X, y=y)

        return self
                                       
    def predict(self, X, return_conf_int=False, alpha=0.05):
        """
        Previsão de regressão voting do target.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.
        return_conf_int : boolean, opcional
            Caso True será retornado, juntamente com a previsão, os
            intervalos de confiança (inferior e superior) baseado somente no
            modelo Sarimax. O valor padrão é False.
        alpha : float, opcional
            Os intervalos de confiança Sarimax das previsões serão (1 - alpha),
            caso return_conf_int = True. O valor padrão é 0.05.
        Returns
        -------
        pandas DataFrame
            Previsão ensemble para a exógena de entrada X.

        """
        y_pred = pd.DataFrame(index=X.index, columns=self.y_train.columns)
        y_pred.loc[:, :] = ((self.xgb_weight * self.Xgb_Regressor.predict(X)) + 
                            (self.sx_weight * self.Sarimax.predict(X)))
        y_pred.loc[:, :] = y_pred.loc[:, :] / (self.xgb_weight + self.sx_weight)
        
        if return_conf_int:
            sarimax_pred = self.Sarimax.predict(X,
                                                return_conf_int=return_conf_int,
                                                alpha=alpha)
            
            
            error = (sarimax_pred.loc[:, self.y_train.columns[0]] - 
                     sarimax_pred.loc[:, 'lower_limit']).copy()
            
            y_pred.loc[:, 'lower_limit'] = y_pred[self.y_train.columns[0]] - error
            y_pred.loc[:, 'upper_limit'] = y_pred[self.y_train.columns[0]] + error
        
        return y_pred