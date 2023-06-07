"""Classe que permite deixar o método SARIMAX no formato sklearn."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Sarimax(BaseEstimator, RegressorMixin):
    """Formatação do método statmodels SARIMAX no estilo sklearn."""

    def __init__(self, p=1, d=0, q=0, P=0, D=0, Q=0, s=0, trend=None,
                 measurement_error=False, time_varying_regression=False,
                 mle_regression=True, simple_differencing=False,
                 enforce_stationarity=True, enforce_invertibility=True,
                 hamilton_representation=False, concentrate_scale=False,
                 trend_offset=1, use_exact_diffuse=False, dates=None,
                 freq=None, missing='none', validate_specification=True,
                 kwargs=None):
        """Inicialização da classe."""
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.trend = trend
        self.me = time_varying_regression
        self.tvr = time_varying_regression
        self.mle_r = mle_regression
        self.sd = simple_differencing
        self.es = enforce_stationarity
        self.ei = enforce_invertibility
        self.hr = hamilton_representation
        self.cs = concentrate_scale
        self.to = trend_offset
        self.ued = use_exact_diffuse
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.vs = validate_specification
        self.kwargs = kwargs

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
            Series contendo a endógena.
        """
        order = (self.p, self.d, self.q)
        seasonal_order = (self.P, self.D, self.Q, self.s)

        endog = y.copy()

        self.y_train = y

        if len(X.columns) != 0:
            exog = X.copy()
        else:
            exog = None

        try:        
            self.sarimax_model = SARIMAX(endog=endog, exog=exog, order=order,
                                         seasonal_order=seasonal_order,
                                         trend=self.trend,
                                         measurement_error=self.me,
                                         time_varying_regression=self.tvr,
                                         mle_regression=self.mle_r,
                                         simple_differencing=self.sd,
                                         enforce_stationarity=self.es,
                                         enforce_invertibility=self.ei,
                                         hamilton_representation=self.hr,
                                         concentrate_scale=self.cs,
                                         trend_offset=self.to,
                                         use_exact_diffuse=self.ued,
                                         dates=self.dates,
                                         freq=self.freq,
                                         missing=self.missing,
                                         validate_specification=self.vs,
                                         kwargs=self.kwargs
                                         )
        
            self.sarimax_model_fit = self.sarimax_model.fit(disp=0)

        except:
            self.sarimax_model = None
            self.sarimax_model_fit = None

        return self

    def predict(self, X):
        """
        Previsão de regressão SARIMAX do target para a exógena X.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.

        Returns
        -------
        y_pred : pandas DataFrame
            Previsão SARIMAX para a exógena de entrada X.

        """
        if len(X.columns) != 0:
            exog = X.copy()
        else:
            exog = None

        if self.sarimax_model_fit is None:
            y_pred = pd.Series(data=np.full(len(X), np.finfo(float).eps),
                               index=X.index)
        else:
            fitted_values = self.sarimax_model_fit.fittedvalues
            y_fitted = fitted_values.loc[fitted_values.index.isin(X.index)]

            X = X[X.index > max(self.y_train.index)]
            if len(X) > 0:
                try:
                    y_forecast = self.sarimax_model_fit.forecast(steps=len(X),
                                                                 exog=exog,
                                                                 index=X.index
                                                                 )
                    if list(y_forecast.index) != list(X.index):
                        raise Exception('Uma ou mais datas no índice de entrada' +
                                        ' não são válidas. As datas devem estar' +
                                        ' no mesmo período e intervalo que as' +
                                        ' usadas no treino. Não é permitido' +
                                        ' pular datas.')
                except:
                    y_forecast = pd.Series(data=np.full(len(X),
                                                        np.finfo(float).eps),
                                           index=X.index)
            else:
                y_forecast = pd.Series(index=X.index)

            y_pred = pd.concat([y_fitted, y_forecast])

            self.y_pred = y_pred
            y_pred = self._forecast_handling()

        return y_pred

    def _forecast_handling(self):
        """
        Tratamento de previsões infinitas pelo SARIMAX.

        Os valores infinitos são substituídos por np.nan e logo em seguida
        é feita uma interpolação linear para substituir os valores np.nan.

        Caso ainda persista nulos, eles serão substituídos por 0.

        Returns
        -------
        y_pred_processed : pandas DataFrame
            Retorna a previsão do SARIMAX tratada.
        """
        y_pred_processed = self.y_pred
        y_pred_processed.loc[y_pred_processed.isin([np.inf, -np.inf]) |
                             (y_pred_processed.abs() > 1e100)] = np.nan

        y_pred_processed = y_pred_processed.interpolate(method='linear')

        # No caso de ainda a previsão tiver nulos.
        y_pred_processed.fillna(np.finfo(float).eps, inplace=True)

        return y_pred_processed
