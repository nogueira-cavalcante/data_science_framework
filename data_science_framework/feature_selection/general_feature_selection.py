"""Função geral para seleção de variáveis."""

# -*- coding: utf-8 -*-


def general_feature_selection(X_train, y_train, lista_features,
                              modo, forma, valor):
    """
    Função geral para seleção de variáveis para classificação e regressão.

    Parameters
    ----------
    X_train : pandas DataFrame
        DataFrame contendo as features de treino.
    y_train : pandas DataFrame
        DataFrame contendo o target de treino.
    lista_features : list
        Lista de strings contendo os nomes das features.
    modo : string
        Modo para a seleção de variáveis ("classification" ou "regression").
    forma : string
        Forma de seleção de variáveis;
            - 'DropFeatures';
            - 'DropConstantFeatures';
            - 'DropDuplicateFeatures';
            - 'DropCorrelatedFeatures';
            - 'SmartCorrelatedSelection';
            - 'SelectByShuffling';
            - 'SelectBySingleFeaturePerformance';
            - 'SelectByTargetMeanPerformance';
            - 'RFE';
            - 'RFA'.
    valor : float
        Usado somente nas formas 'DropConstantFeatures' (parâmetro tol) e
        e 'DropCorrelatedFeatures' parâmetro (threshold).

    Raises
    ------
    Exception
        - Quando o valor em "modo" informado não se encontra implementado ou
          não existe;
        - Quando o valor em "forma" informado não se encontra implementado ou
          não existe.

    Returns
    -------
    train_t : array-like
        Dados tratados com a seleção de variáveis escolhida.
    features_selecionadas : list
        Lista de variáveis selecionadas.

    """
    if modo == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor()
        scoring = 'r2'

    elif modo == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier()
        scoring = 'roc_auc_score'

    else:
        raise Exception('O modo informado é inválido (' + modo + ').')

    if forma == 'DropFeatures':
        from feature_engine.selection import DropFeatures
        transformer = DropFeatures(features_to_drop=lista_features)
        transformer.fit(X_train)

    elif forma == 'DropConstantFeatures':
        from feature_engine.selection import DropConstantFeatures
        transformer = DropConstantFeatures(tol=valor,
                                           variables=lista_features,
                                           missing_values='ignore')
        transformer.fit(X_train)

    elif forma == 'DropDuplicateFeatures':
        from feature_engine.selection import DropDuplicateFeatures
        transformer = DropDuplicateFeatures(variables=lista_features,
                                            missing_values='ignnore')
        transformer.fit(X_train)

    elif forma == 'DropCorrelatedFeatures':
        from feature_engine.selection import DropCorrelatedFeatures
        transformer = DropCorrelatedFeatures(variables=lista_features,
                                             method='pearson',
                                             threshold=valor)
        transformer.fit(X_train)

    elif forma == 'SmartCorrelatedSelection':
        from feature_engine.selection import SmartCorrelatedSelection
        transformer = SmartCorrelatedSelection(variables=lista_features,
                                               method="pearson",
                                               threshold=0.8,
                                               missing_values="raise",
                                               selection_method="variance",
                                               estimator=None)
        transformer.fit(X_train)

    elif forma == 'SelectByShuffling':
        from feature_engine.selection import SelectByShuffling
        transformer = SelectByShuffling(variables=lista_features,
                                        estimator=estimator,
                                        scoring=scoring, random_state=0)
        transformer.fit(X_train, y_train)

    elif forma == 'SelectBySingleFeaturePerformance':
        from feature_engine.selection import SelectBySingleFeaturePerformance
        transformer = SelectBySingleFeaturePerformance(estimator=estimator,
                                                       scoring=scoring,
                                                       threshold=0.01)
        transformer.fit(X_train, y_train)

    elif forma == 'RFE':
        from feature_engine.selection import RecursiveFeatureElimination
        transformer = RecursiveFeatureElimination(estimator=estimator,
                                                  variables=lista_features,
                                                  scoring=scoring,
                                                  cv=3)
        transformer.fit(X_train, y_train)
    elif forma == 'RFA':
        from feature_engine.selection import RecursiveFeatureAddition
        transformer = RecursiveFeatureAddition(estimator=estimator,
                                               variables=lista_features,
                                               scoring=scoring,
                                               cv=3)
        transformer.fit(X_train, y_train)
    else:
        raise Exception('A forma informada é inválida (' + forma + ').')

    train_t = transformer.transform(X_train)
    features_selecionadas = list(train_t.columns)

    return train_t, features_selecionadas
