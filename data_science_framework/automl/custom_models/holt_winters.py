#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classe que permite deixar o método Holt-Winters no formato sklearn."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWinters(BaseEstimator, RegressorMixin):
    """Formatação do método statmodels Holt-Winters no estilo sklearn."""

    def __init__(self, trend=None, damped_trend=False, seasonal=None,
                 seasonal_periods=None, initialization_method=None,
                 initial_level=None, initial_trend=None, initial_seasonal=None,
                 use_boxcox=None, bounds=None, dates=None, freq=None,
                 missing='none', smoothing_level=None, smoothing_trend=None,
                 smoothing_seasonal=None, optimized=True, remove_bias=False,
                 minimize_kwargs=None, use_brute=True):
        """Inicialização da classe."""
        self.trend = trend
        self.d_t = damped_trend
        self.seasonal = seasonal
        self.s_p = seasonal_periods
        self.i_m = initialization_method
        self.i_l = initial_level
        self.i_t = initial_trend
        self.i_s = initial_seasonal
        self.use_b = use_boxcox
        self.b = bounds
        self.dates = dates
        self.freq = freq
        self.m = missing
        self.s_l = smoothing_level
        self.s_t = smoothing_trend
        self.s_s = smoothing_seasonal
        self.o = optimized
        self.r_b = remove_bias
        self.m_k = minimize_kwargs
        self.u_b = use_brute

    def fit(self, X, y):
        """
        Construção do modelo Holt-Winters para um conjunto de treino (X, y).

        Os índices devem estar no formato datetime. Por exemplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.
        y : pandas DataFrame
            DataFrame contendo a endógena.
        """
        self.y_train = y
        ES = ExponentialSmoothing

        self.hw_model = ES(endog=y,
                           trend=self.trend,
                           damped_trend=self.d_t,
                           seasonal=self.seasonal,
                           seasonal_periods=self.s_p,
                           initialization_method=self.i_m,
                           initial_level=self.i_l,
                           initial_trend=self.i_t,
                           initial_seasonal=self.i_s,
                           use_boxcox=self.use_b,
                           bounds=self.b,
                           dates=self.dates,
                           freq=self.freq,
                           missing=self.m
                           )

        try:
            self.hw_model_fit = self.hw_model.fit(
                                                  smoothing_level=self.s_l,
                                                  smoothing_trend=self.s_t,
                                                  smoothing_seasonal=self.s_s,
                                                  optimized=self.o,
                                                  remove_bias=self.r_b,
                                                  minimize_kwargs=self.m_k,
                                                  use_brute=self.u_b
                                                 )

        except:
            self.hw_model = None
            self.hw_model_fit = None

        return self

    def predict(self, X):
        """
        Previsão de regressão Holt-Winters do target.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.

        Returns
        -------
        pandas DataFrame
            Previsão Holt-Winters para a exógena de entrada X.

        """
        if self.hw_model_fit is None:
            y_pred = pd.Series(data=np.full(len(X), np.finfo(float).eps),
                               index=X.index)
        else:
            fitted_values = self.hw_model_fit.fittedvalues
            y_fitted = fitted_values.loc[fitted_values.index.isin(X.index)]

            X = X[X.index > max(self.y_train.index)]
            if len(X) > 0:
                y_forecast = self.hw_model_fit.forecast(len(X))
                if list(y_forecast.index) != list(X.index):
                    raise Exception('Uma ou mais datas no índice de entrada' +
                                    ' não são válidas. As datas devem estar' +
                                    ' no mesmo período e intervalo que as' +
                                    ' usadas no treino. Não é permitido' +
                                    ' pular datas.')
            else:
                y_forecast = pd.Series(index=X.index)

            y_pred = pd.concat([y_fitted, y_forecast])

            self.y_pred = y_pred
            y_pred = self._forecast_handling()

            y_pred = pd.DataFrame(data=y_pred.values,
                                  columns=self.y_train.columns,
                                  index=y_pred.index)

        return y_pred

    def _forecast_handling(self):
        """
        Tratamento de previsões infinitas pelo Holt_Winter.

        Os valores infinitos são substituídos por np.nan e logo em seguida
        é feita uma interpolação linear para substituir os valores np.nan.

        Caso ainda persista nulos, eles serão substituídos por 0.

        Returns
        -------
        pandas Series
            Previsão do Holt-Winters tratada.
        """
        y_pred_processed = self.y_pred
        y_pred_processed.loc[y_pred_processed.isin([np.inf, -np.inf]) |
                             (y_pred_processed.abs() > 1e100)] = np.nan

        y_pred_processed = y_pred_processed.interpolate(method='linear')

        # No caso de ainda a previsão tiver nulos.
        y_pred_processed.fillna(np.finfo(float).eps, inplace=True)

        return y_pred_processed
