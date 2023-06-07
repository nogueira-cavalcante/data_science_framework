# -*- coding: utf-8 -*-

"""Classe que permite deixar o método Neural Prophet no formato sklearn."""

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from neuralprophet import NeuralProphet


class Neural_Prophet(BaseEstimator, RegressorMixin):
    """Formatação do método Neural Prophet no estilo sklearn."""

    def __init__(self, growth='linear', changepoints=None, n_changepoints=10,
                 changepoints_range=0.9, trend_reg=0,
                 trend_reg_threshold=False, yearly_seasonality='auto',
                 weekly_seasonality='auto', daily_seasonality='auto',
                 seasonality_mode='additive', seasonality_reg=0, n_forecasts=1,
                 n_lags=0, num_hidden_layers=0, d_hidden=None,
                 ar_sparsity=None, learning_rate=None, epochs=None,
                 batch_size=None, loss_func='Huber', optimizer='AdamW',
                 train_speed=None, normalize='auto', impute_missing=True):
        """Inicialização da classe."""
        self.g = growth
        self.c = changepoints
        self.n_c = n_changepoints
        self.c_r = changepoints_range
        self.t_r = trend_reg
        self.t_r_t = trend_reg_threshold
        self.y_s = yearly_seasonality
        self.w_s = weekly_seasonality
        self.d_s = daily_seasonality
        self.s_m = seasonality_mode
        self.s_r = seasonality_reg
        self.n_f = n_forecasts
        self.n_l = n_lags
        self.n_h_l = num_hidden_layers
        self.d_h = d_hidden
        self.a_s = ar_sparsity
        self.l_r = learning_rate
        self.e = epochs
        self.b_s = batch_size
        self.l_f = loss_func
        self.o = optimizer
        self.t_s = train_speed
        self.n = normalize
        self.i_m = impute_missing

    def fit(self, X, y):
        """
        Construção do modelo Neural Prophet para um conjunto de treino (X, y).

        Os índices devem estar no formato datetime. Por exemplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Parameters
        ----------
        X : (pandas DataFrame)
            DataFrame contendo as exógenas..
        y : (pandas DataFrame)
            DataFrame contendo a endógena.
        """
        self.y_train = y.copy()

        self.np_model = NeuralProphet(growth=self.g,
                                      changepoints=self.c,
                                      n_changepoints=self.n_c,
                                      changepoints_range=self.c_r,
                                      trend_reg=self.t_r,
                                      trend_reg_threshold=self.t_r_t,
                                      yearly_seasonality=self.y_s,
                                      weekly_seasonality=self.w_s,
                                      daily_seasonality=self.d_s,
                                      seasonality_mode=self.s_m,
                                      seasonality_reg=self.s_r,
                                      n_forecasts=self.n_f,
                                      n_lags=self.n_l,
                                      num_hidden_layers=self.n_h_l,
                                      d_hidden=self.d_h,
                                      ar_sparsity=self.a_s,
                                      learning_rate=self.l_r,
                                      epochs=self.e,
                                      batch_size=self.b_s,
                                      loss_func=self.l_f,
                                      optimizer=self.o,
                                      train_speed=self.t_s,
                                      normalize=self.n,
                                      impute_missing=self.i_m)

        if len(X.columns) > 0:
            for column in X.columns:
                self.np_model = self.np_model.add_future_regressor(name=column)

        train = X.copy()
        train.loc[:, 'y'] = y[y.columns[0]].copy()

        if train.index.name:
            index_name = train.index.name
        else:
            index_name = 'index'

        train = train.reset_index().rename(columns={index_name: 'ds'})
        self.train = train.copy()

        self.np_model_fit = self.np_model.fit(df=train,
                                              freq=X.index.freqstr,
                                              epochs=None,
                                              validate_each_epoch=False,
                                              valid_p=0.2,
                                              progress_bar=True,
                                              plot_live_loss=False)

        return self

    def predict(self, X):
        """
        Previsão de regressão Neural Prophet do target para a exógena X.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.

        Returns
        -------
        pandas DataFrame
            Previsão Neural Prophet para a exógena de entrada X.

        """
        if X.index.name:
            index_name = X.index.name
        else:
            index_name = 'index'

        if len(X.columns) > 0:
            regs_df = X.reset_index().copy()
            regs_df.rename(columns={index_name: 'ds'}, inplace=True)
        else:
            regs_df = None

        prediction = self.np_model.make_future_dataframe(df=self.train,
                                                         regressors_df=regs_df,
                                                         periods=len(X),
                                                         n_historic_predictions=len(self.train))

        y_pred = self.np_model.predict(df=prediction)[['ds', 'yhat1']]
        y_pred.drop_duplicates('ds', inplace=True)

        y_pred.set_index('ds', inplace=True)
        y_pred.index.name = index_name
        y_pred.columns = self.y_train.columns
        y_pred = y_pred[y_pred.index.isin(X.index)]

        return y_pred
