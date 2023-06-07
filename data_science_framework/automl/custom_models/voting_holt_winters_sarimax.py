# -*- coding: utf-8 -*-

"""Classe voting unificando os métodos Sarimax e Holt-Winters."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from .sarimax import Sarimax
from .holt_winters import HoltWinters

class Voting_HoltWinters_Sarimax(BaseEstimator, RegressorMixin):
    """Classe ensemble unificando os métodos Sarimax e Holt-Winters."""
    
    def __init__(self, hw_trend=None, hw_damped_trend=False, hw_seasonal=None,
                 hw_seasonal_periods=None, hw_initialization_method=None,
                 hw_initial_level=None, hw_initial_trend=None,
                 hw_initial_seasonal=None, hw_use_boxcox=None, hw_bounds=None,
                 hw_dates=None, hw_freq=None, hw_missing='none',
                 hw_smoothing_level=None, hw_smoothing_trend=None,
                 hw_smoothing_seasonal=None, hw_optimized=True,
                 hw_remove_bias=False, hw_minimize_kwargs=None,
                 hw_use_brute=True, hw_weight=0.5, sx_start_p=2, sx_d=None,
                 sx_start_q=2, sx_max_p=5, sx_max_d=2, sx_max_q=5,
                 sx_start_P=1, sx_D=None, sx_start_Q=1, sx_max_P=2, sx_max_D=1,
                 sx_max_Q=2, sx_max_order=5, sx_m=1, sx_seasonal=True,
                 sx_stationary=False, sx_information_criterion='aic',
                 sx_alpha=0.05, sx_test='kpss', sx_seasonal_test='ocsb',
                 sx_stepwise=True, sx_n_jobs=1, sx_start_params=None,
                 sx_trend=None, sx_method='lbfgs', sx_maxiter=50,
                 sx_offset_test_args=None, sx_seasonal_test_args=None,
                 sx_suppress_warnings=True, sx_error_action='trace',
                 sx_trace=False, sx_random=False, sx_random_state=None,
                 sx_n_fits=10, sx_return_valid_fits=False,
                 sx_out_of_sample_size=0, sx_scoring='mse',
                 sx_scoring_args=None, sx_with_intercept='auto',
                 sx_sarimax_kwargs=None, sx_weight=0.5):
        """Inicialização da classe."""
        
        self.HoltWinters = HoltWinters(trend=hw_trend,
                                       damped_trend=hw_damped_trend,
                                       seasonal=hw_seasonal,
                                       seasonal_periods=hw_seasonal_periods,
                                       initialization_method=hw_initialization_method,
                                       initial_level=hw_initial_level,
                                       initial_trend=hw_initial_trend,
                                       initial_seasonal=hw_initial_seasonal,
                                       use_boxcox=hw_use_boxcox,
                                       bounds=hw_bounds, dates=hw_dates,
                                       freq=hw_freq, missing=hw_missing,
                                       smoothing_level=hw_smoothing_level,
                                       smoothing_trend=hw_smoothing_trend,
                                       smoothing_seasonal=hw_smoothing_seasonal,
                                       optimized=hw_optimized,
                                       remove_bias=hw_remove_bias,
                                       minimize_kwargs=hw_minimize_kwargs,
                                       use_brute=hw_use_brute)
                                       
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
                               scoring=sx_scoring,
                               scoring_args=sx_scoring_args,
                               with_intercept=sx_with_intercept,
                               sarimax_kwargs=sx_sarimax_kwargs)
        
        self.hw_weight = hw_weight
        self.sx_weight = sx_weight
        
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
                                       
        self.HoltWinters.fit(X=X, y=y)
        self.Sarimax.fit(X=X, y=y)

        return self
                                       
    def predict(self, X):
        """
        Previsão de regressão voting do target.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.

        Returns
        -------
        pandas DataFrame
            Previsão voting para a exógena de entrada X.

        """
        y_pred = pd.DataFrame(index=X.index, columns=self.y_train.columns)
        y_pred.loc[:, :] = ((self.hw_weight * self.HoltWinters.predict(X)) +
                            (self.sx_weight * self.Sarimax.predict(X)))
        y_pred.loc[:, :] = y_pred.loc[:, :] / (self.hw_weight + self.sx_weight)
        
        return y_pred