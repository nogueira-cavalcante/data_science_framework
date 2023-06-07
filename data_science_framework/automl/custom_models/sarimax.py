# -*- coding: utf-8 -*-

"""Classe que permite deixar o método SARIMAX no formato sklearn."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from pmdarima.arima import auto_arima


class Sarimax(BaseEstimator, RegressorMixin):
    """Formatação do método statmodels SARIMAX no estilo sklearn."""

    def __init__(self, start_p=2, d=None, start_q=2, max_p=5,
                 max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
                 max_D=1, max_Q=2, max_order=5, m=1, seasonal=True,
                 stationary=False, information_criterion='aic',
                 alpha=0.05, test='kpss', seasonal_test='ocsb',
                 stepwise=True, n_jobs=1, start_params=None, trend=None,
                 method='lbfgs', maxiter=50, offset_test_args=None,
                 seasonal_test_args=None, suppress_warnings=True,
                 error_action='trace', trace=False, random=False,
                 random_state=None, n_fits=10, return_valid_fits=False,
                 out_of_sample_size=0, scoring='mse', scoring_args=None,
                 with_intercept='auto', sarimax_kwargs=None):
        """Inicialização da classe."""
        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.max_order = max_order
        self.m = m
        self.seasonal = seasonal
        self.stationary = stationary
        self.info_c = information_criterion
        self.alpha = alpha
        self.test = test
        self.s_test = seasonal_test
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.maxiter = maxiter
        self.ota = offset_test_args
        self.sta = seasonal_test_args
        self.sw = suppress_warnings
        self.ea = error_action
        self.trace = trace
        self.random = random
        self.random_s = random_state
        self.n_fits = n_fits
        self.rvf = return_valid_fits
        self.ooss = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.with_i = with_intercept
        self.sarimax_k = sarimax_kwargs

    def fit(self, X, y):
        """
        Construção do modelo SARIMAX para um conjunto de treino (X, y).

        Os índices devem estar no formato datetime. Por exemplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Parameters
        ----------
        X : (pandas DataFrame)
            DataFrame contendo as exógenas..
        y : (pandas DataFrame)
            DataFrame contendo a endógena.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        endog = y.copy()

        if len(X.columns) != 0:
            exog = X.copy()
        else:
            exog = None

        try:
            self.sarimax_model = auto_arima(y=endog,
                                            X=exog,
                                            start_p=self.start_p,
                                            d=self.d,
                                            start_q=self.start_q,
                                            max_p=self.max_p,
                                            max_d=self.max_d,
                                            max_q=self.max_q,
                                            start_P=self.start_P,
                                            D=self.D,
                                            start_Q=self.start_Q,
                                            max_P=self.max_P,
                                            max_D=self.max_D,
                                            max_Q=self.max_Q,
                                            max_order=self.max_order,
                                            m=int(self.m),
                                            seasonal=self.seasonal,
                                            stationary=self.stationary,
                                            information_criterion=self.info_c,
                                            alpha=self.alpha,
                                            test=self.test,
                                            seasonal_test=self.s_test,
                                            stepwise=self.stepwise,
                                            n_jobs=self.n_jobs,
                                            start_params=self.start_params,
                                            trend=self.trend,
                                            method=self.method,
                                            maxiter=self.maxiter,
                                            offset_test_args=self.ota,
                                            seasonal_test_args=self.sta,
                                            suppress_warnings=self.sw,
                                            error_action=self.ea,
                                            trace=self.trace,
                                            random=self.random,
                                            random_state=self.random_s,
                                            n_fits=self.n_fits,
                                            return_valid_fits=self.rvf,
                                            out_of_sample_size=int(self.ooss),
                                            scoring=self.scoring,
                                            scoring_args=self.scoring_args,
                                            with_intercept=self.with_i,
                                            sarimax_kwargs=self.sarimax_k)
        except:
            self.sarimax_model = None

        return self

    def predict(self, X, return_conf_int=False, alpha=0.05):
        """
        Previsão de regressão SARIMAX do target para a exógena X.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.
        return_conf_int : boolean, opcional
            Caso True será retornado, juntamente com a previsão, os
            intervalos de confiança (inferior e superior). O valor padrão
            é False.
        alpha : float, opcional
            Os intervalos de confiança das previsões serão (1 - alpha), caso
            return_conf_int = True. O valor padrão é 0.05.
        Returns
        -------
        pandas DataFrame
            Previsão SARIMAX para a exógena de entrada X e os intervalos
            mínimo e máximo de confiança (caso return_conf_int seja True).
        """
        if self.sarimax_model is None:
            y_pred = pd.Series(data=np.full(len(X), np.finfo(float).eps),
                               index=X.index)
            
            y_conf_int = pd.DataFrame(index=X.index)
            y_conf_int['lower_limit'] = np.full(len(X), np.finfo(float).eps)
            y_conf_int['upper_limit'] = np.full(len(X), np.finfo(float).eps)

        else:

            (fitted_values,
             fitted_conf_int) = self.sarimax_model.predict_in_sample(X=self.X_train,
                                                                     alpha=alpha,
                                                                     return_conf_int=True)

            fitted_values = pd.Series(data=fitted_values,
                                      index=self.y_train.index)
            fitted_conf_int = pd.DataFrame(data=fitted_conf_int,
                                           columns=['lower_limit', 'upper_limit'],
                                           index=self.y_train.index)

            y_fitted = fitted_values.loc[fitted_values.index.isin(X.index)]
            conf_int_fitted = fitted_conf_int.loc[fitted_conf_int.index.isin(X.index)]

            X = X[X.index > max(self.y_train.index)]
            if len(X) > 0:

                try:
                    (y_forecast,
                     conf_int_forecast) = self.sarimax_model.predict(n_periods=len(X),
                                                                     X=X,
                                                                     alpha=alpha,
                                                                     return_conf_int=True)                    
                except:
                    y_forecast = np.full(len(X), np.finfo(float).eps)
                    conf_int_forecast = np.array([y_forecast,
                                                  y_forecast]).transpose()

                y_forecast = pd.Series(data=y_forecast, index=X.index)
                conf_int_forecast = pd.DataFrame(data=conf_int_forecast,
                                                 columns=['lower_limit', 'upper_limit'],
                                                 index=X.index)

                if list(y_forecast.index) != list(X.index):
                    raise Exception('Uma ou mais datas no índice de ' +
                                    'entrada não são válidas. As datas ' +
                                    'devem estar no mesmo período e ' +
                                    'intervalo que as usadas no treino. ' +
                                    'Não é permitido pular datas.')

            else:
                y_forecast = pd.Series(index=X.index)
                conf_int_forecast = pd.DataFrame(columns=['lower_limit', 'upper_limit'],
                                                 index=X.index)

            y_pred = pd.concat([y_fitted, y_forecast])
            conf_int_pred = pd.concat([conf_int_fitted, conf_int_forecast])

            self.y_pred = y_pred
            self.conf_int_pred = conf_int_pred
            
            (y_pred,
             conf_int_pred) = self._forecast_handling()

        y_pred = pd.DataFrame(data=y_pred.values,
                              columns=self.y_train.columns,
                              index=y_pred.index)
        
        if return_conf_int:
            y_pred = pd.merge(y_pred,
                              conf_int_pred,
                              how='left',
                              left_index=True,
                              right_index=True)

        return y_pred

    def _forecast_handling(self):
        """
        Tratamento de previsões infinitas pelo SARIMAX.

        Os valores infinitos são substituídos por np.nan e logo em seguida
        é feita uma interpolação linear para substituir os valores np.nan.

        Caso ainda persista nulos, eles serão substituídos por 0.

        Returns
        -------
        pandas Series
            Retorna a previsão do SARIMAX tratada.
        """
        y_pred_processed = self.y_pred
        y_pred_processed.loc[y_pred_processed.isin([np.inf, -np.inf]) |
                             (y_pred_processed.abs() > 1e100)] = np.nan
        y_pred_processed = y_pred_processed.interpolate(method='linear')
        y_pred_processed = y_pred_processed.interpolate(method='linear')
             
        conf_int_pred_processed = self.conf_int_pred
        conf_int_pred_processed.loc[conf_int_pred_processed['lower_limit'].isin([np.inf, -np.inf]) |
                                    (conf_int_pred_processed['lower_limit'].abs() > 1e100), 'lower_limit'] = np.nan
        conf_int_pred_processed.loc[conf_int_pred_processed['upper_limit'].isin([np.inf, -np.inf]) |
                                    (conf_int_pred_processed['upper_limit'].abs() > 1e100), 'upper_limit'] = np.nan
        
        conf_int_pred_processed['lower_limit'] = conf_int_pred_processed['lower_limit'].interpolate(method='linear')
        conf_int_pred_processed['upper_limit'] = conf_int_pred_processed['upper_limit'].interpolate(method='linear')


        # No caso de ainda a previsão tiver nulos.
        y_pred_processed.fillna(np.finfo(float).eps, inplace=True)
        conf_int_pred_processed.fillna(np.finfo(float).eps, inplace=True)
        
        return y_pred_processed, conf_int_pred_processed
