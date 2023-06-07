# -*- coding: utf-8 -*-

"""Classe que permite deixar o método XGBRegressor no formato sklearn."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor


class Xgb_Regressor(BaseEstimator, RegressorMixin):
    """Formatação do método statmodels SARIMAX no estilo sklearn."""

    def __init__(self, n_estimators=100, max_depth=None, learning_rate=None,
                 verbosity=None, objective='reg:squarederror', booster=None,
                 tree_method=None, n_jobs=None, gamma=None, eta=0.3,
                 min_child_weight=None, max_delta_step=None, subsample=None,
                 colsample_bytree=None, colsample_bylevel=None,
                 colsample_bynode=None, reg_alpha=None, reg_lambda=None,
                 scale_pos_weight=None, base_score=None, random_state=None,
                 missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None,
                 importance_type='gain', gpu_id=None,
                 validate_parameters=None):
        """Inicialização da classe."""
        self.n_e = n_estimators
        self.m_d = max_depth
        self.l_r = learning_rate
        self.v = verbosity
        self.o = objective
        self.b = booster
        self.t_m = tree_method
        self.n_j = n_jobs
        self.g = gamma
        self.e = eta
        self.m_c_w = min_child_weight
        self.m_d_s = max_delta_step
        self.s = subsample
        self.c_bytree = colsample_bytree
        self.c_bylevel = colsample_bylevel
        self.c_bynode = colsample_bynode
        self.r_a = reg_alpha
        self.r_l = reg_lambda
        self.s_p_w = scale_pos_weight
        self.b_s = base_score
        self.r_s = random_state
        self.m = missing
        self.n_p_t = num_parallel_tree
        self.m_c = monotone_constraints
        self.i_c = interaction_constraints
        self.i_t = importance_type
        self.g_i = gpu_id
        self.v_p = validate_parameters

    def fit(self, X, y):
        """
        Construção do modelo XGBRegressor para um conjunto de treino (X, y).

        Os índices devem estar no formato datetime. Por exemplo,
        df.index = pd.DatetimeIndex(df.index).to_period(period).

        Parameters
        ----------
        X : (pandas DataFrame)
            DataFrame contendo as exógenas..
        y : (pandas DataFrame)
            DataFrame contendo a endógena.
        """
        self.y_train = y

        self.xgb_model = XGBRegressor(n_estimators=self.n_e,
                                      max_depth=self.m_d,
                                      learning_rate=self.l_r,
                                      verbosity=self.v,
                                      objective=self.o,
                                      booster=self.b,
                                      tree_method=self.t_m,
                                      n_jobs=self.n_j,
                                      gamma=self.g,
                                      eta=self.e,
                                      min_child_weight=self.m_c_w,
                                      max_delta_step=self.m_d_s,
                                      subsample=self.s,
                                      colsample_bytree=self.c_bytree,
                                      colsample_bylevel=self.c_bylevel,
                                      colsample_bynode=self.c_bynode,
                                      reg_alpha=self.r_a,
                                      reg_lambda=self.r_l,
                                      scale_pos_weight=self.s_p_w,
                                      base_score=self.b_s,
                                      random_state=self.r_s,
                                      missing=self.m,
                                      num_parallel_tree=self.n_p_t,
                                      monotone_constraints=self.m_c,
                                      interaction_constraints=self.i_c,
                                      importance_type=self.i_t,
                                      gpu_id=self.g_i
                                      )
        if len(X.columns) != 0:
            exog = X.copy()
        else:
            exog = None

        self.xgb_model.fit(X=exog,
                           y=y,
                           sample_weight=None,
                           base_margin=None,
                           eval_set=None,
                           eval_metric=None,
                           early_stopping_rounds=None,
                           verbose=True,
                           sample_weight_eval_set=None,
                           base_margin_eval_set=None,
                           feature_weights=None,
                           callbacks=None)

        return self

    def predict(self, X):
        """
        Previsão de regressão XGBRegressor do target para a exógena X.

        Parameters
        ----------
        X : pandas DataFrame
            DataFrame contendo as exógenas.

        Returns
        -------
        pandas DataFrame
            Previsão XGBRegressor para a exógena de entrada X.

        """
        if len(X.columns) != 0:
            exog = X.copy()
        else:
            exog = None

        y_pred = self.xgb_model.predict(X=exog,
                                        output_margin=False,
                                        ntree_limit=None,
                                        validate_features=True,
                                        base_margin=None,
                                        iteration_range=None)

        y_pred = pd.DataFrame(data=y_pred,
                              columns=self.y_train.columns,
                              index=X.index)

        return y_pred
