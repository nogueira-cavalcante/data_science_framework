"""Interpretação dos dados usando lime."""

# -*- coding: utf-8 -*-


def ri_lime_tabular(X_train, X_test, modelo, index_sample,
                    class_names, modo):
    """
    Interpretação dos dados tabulares usando lime.

    Parameters
    ----------
    X_train : pandas DataFrame
        DataFrame contendo as features de treino.
    X_test : pandas DataFrame
        DataFrame contendo o target de treino.
    modelo : object
        Objeto contendo o modelo treinado.
    index_sample : list ou index-like
        Índices indicando as amostras para o uso do lime.
    class_names : list
        Lista de nomes de classes, ordenado de acordo com o classificador
        usado.
    modo : string
        "classication" ou "regression"

    Returns
    -------
    exp : objetc
        Objeto lime.

    """
    from lime import lime_tabular

    cols_values = X_train.columns.values
    explainer = lime_tabular.LimeTabularExplainer(X_train.values,
                                                  feature_names=cols_values,
                                                  class_names=class_names,
                                                  verbose=True,
                                                  mode=modo)

    if modo == 'regression':
        exp = explainer.explain_instance(X_test.iloc[index_sample],
                                         modelo.predict,
                                         num_features=len(X_train.columns))
    else:
        exp = explainer.explain_instance(X_test.iloc[index_sample],
                                         modelo.predict_proba,
                                         num_features=len(X_train.columns))
    exp.show_in_notebook(show_table=True)
    exp.as_list()

    return exp
