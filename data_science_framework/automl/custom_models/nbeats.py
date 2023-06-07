# -*- coding: utf-8 -*-

"""Classe que permite deixar o método NBeats no formato sklearn."""

import pandas as pd
import tensorflow
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from nbeats_keras.model import NBeatsNet as NBeatsKeras


class NBeats_Keras(BaseEstimator, RegressorMixin):
    """Formatação do método NBeats no estilo sklearn."""

    def __init__(self, nb_blocks_per_stack=2, thetas_dim=(4, 8),
                 share_weights_in_stack=True, hidden_layer_units=10, epochs=20,
                 batch_size=2, loss='mse', optimizer='nadam', verbose=False,
                 random_state=10):
        """Inicialização da classe."""
        self.nb = nb_blocks_per_stack
        self.t_d = thetas_dim
        self.s_w = share_weights_in_stack
        self.h_l_u = hidden_layer_units
        self.e = epochs
        self.o = optimizer
        self.loss = loss
        self.b_s = batch_size
        self.v = verbose
        self.r_s = random_state

    def fit(self, X, y):
        """
        Construção do modelo NBeats para um conjunto de treino (X, y).

        Os índices devem estar no formato datetime. Por exemplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Parameters
        ----------
        X : (pandas DataFrame)
            DataFrame contendo as exógenas..
        y : (pandas DataFrame)
            DataFrame contendo a endógena.
        """
        warnings.filterwarnings(action='ignore', message='Setting attributes')

        self.y_train = y.copy()

        tensorflow.random.set_seed(self.r_s)

        self.nbeat_model = NBeatsKeras(backcast_length=X.shape[1],
                                       forecast_length=y.shape[1],
                                       stack_types=(NBeatsKeras.GENERIC_BLOCK,
                                                    NBeatsKeras.GENERIC_BLOCK),
                                       nb_blocks_per_stack=self.nb,
                                       thetas_dim=self.t_d,
                                       share_weights_in_stack=self.s_w,
                                       hidden_layer_units=self.h_l_u
                                       )

        # Definition of the objective function and the optimizer.
        self.nbeat_model.compile(loss=self.loss, optimizer=self.o)

        X_reshaped = X.values.reshape(X.shape[0], X.shape[1], 1)
        y_reshaped = y.values.reshape(y.shape[0], y.shape[1], 1)

        self.nbeat_model.fit(X_reshaped,
                             y_reshaped,
                             epochs=self.e,
                             batch_size=self.b_s,
                             verbose=self.v
                             )

        return self

    def predict(self, X):
        """
        Previsão de regressão NBeats do target para a exógena X.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.

        Returns
        -------
        pandas DataFrame
            Previsão NBeats para a exógena de entrada X.

        """
        X_reshaped = X.values.reshape(X.shape[0], X.shape[1], 1)

        y_pred = self.nbeat_model.predict(X_reshaped)

        y_pred = pd.DataFrame(data=y_pred.reshape(y_pred.shape[0],
                                                  y_pred.shape[1]),
                              columns=self.y_train.columns,
                              index=X.index)

        return y_pred
