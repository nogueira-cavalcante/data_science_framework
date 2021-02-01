import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Sarimax(BaseEstimator, RegressorMixin):

    """
    Classe que permite deixar o método statmodels SARIMAX no formato de um
    estimador sklearn.
    """

    def __init__(self, p=1, d=0, q=0, P=0, D=0, Q=0, s=0, trend=None, me=False,
                 tvr=False, mle_r=True, sd=False, es=True, ei=True, hr=False,
                 cs=False, to=1, ued=False, dates=None, freq=None,
                 missing='none', kwargs={}):

        """
        Inicialização da classe.
        """

        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.trend = trend
        self.me = me
        self.tvr = tvr
        self.mle_r = mle_r
        self.sd = sd
        self.es = es
        self.ei = ei
        self.hr = hr
        self.cs = cs
        self.to = to
        self.ued = ued
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Construção do modelo SARIMAX para um conjunto de treino (X, y).
        Idealmente os índices devem estar no formato datetime. Por exmplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Args:
            X (pandas DataFrame): DataFrame contendo as exógenas.
            y (pandas Series): Series contendo a endógena.

        """

        order = (self.p, self.d, self.q)
        seasonal_order = (self.P, self.D, self.Q, self.s)

        self.sarimax_model = SARIMAX(endog=y, exog=X, order=order,
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
                                     kwargs=self.kwargs
                                     )
        self.sarimax_model_fit = self.sarimax_model.fit(disp=0)

        return self

    def predict(self, X):
        """
        Previsão de regressão SARIMAX do target para a exógena X.

        Args:
            X (pandas DataFrame): DataFrame contendo as exógenas.

        Return:
            Previsão SARIMAX para a exógena de entrada X.

        """

        y_pred = self.sarimax_model_fit.forecast(steps=len(X),
                                                 exog=X,
                                                 index=X.index)
        self.y_pred = y_pred
        y_pred = self._forecast_handling()

        return y_pred

    def _forecast_handling(self):
        """
        Tratamento de previsões infinitos pelo SARIMAX. Esse valores são
        substituídos por np.nan e logo em seguida é feita uma
        interpolação linear para substituir os valores np.nan.

        Return:
            Retorna a previsão do SARIMAX tratada.

        """

        y_pred_processed = self.y_pred
        y_pred_processed.loc[y_pred_processed.isin([np.inf, -np.inf])] = np.nan
        y_pred_processed = y_pred_processed.interpolate(method='linear')

        return y_pred_processed
