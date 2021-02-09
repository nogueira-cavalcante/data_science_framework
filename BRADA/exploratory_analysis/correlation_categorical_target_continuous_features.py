"""Análise de correlação entre variáveis categóricas e contínuas."""


def correlation_categorical_target_continuous_features(df, target_col,
                                                       features_cols=None,
                                                       fig_size_x=10,
                                                       fig_size_y=5):
    """
    Análise de correlação para target categórico e features contínuas.

    Esta função utiliza o ExtraTree para a análise de correlação.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo o target e as features.
    target_col : string
        Nome da coluna target.
    features_cols : list, optional
        lista de strings contendo os nomes das features. O valor padrão é None.
        Caso esse parâmetro não seja informado, serão consideradas as
        demais colunas como features.
    fig_size_x : integer, optional
        Tamanho do eixo x para o gráfico. O valor padrão é 15.
    fig_size_y : integer, optional
        Tamanho do eixo y para o gráfico. O valor padrão é 5.

    Returns
    -------
    Gráfico mostrando a análise de correlação entre as variáveis features com
    o target, através do ExtraTree.

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import ExtraTreesClassifier

    if target_col is None:
        target_col = list(df.columns).remove(target_col)

    df_temp = df[[target_col] + features_cols].copy()

    X = df_temp[features_cols]
    y = df_temp[target_col]

    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = []
    for tree in forest.estimators_:
        std.append(tree.feature_importances_)
    std = np.std(std)

    df_importances = pd.DataFrame(columns=['features',
                                           'feature_importances',
                                           'std_feature_importances'])
    df_importances['features'] = df_temp[features_cols].columns
    df_importances['feature_importances'] = importances
    df_importances['std_feature_importances'] = std

    df_importances.sort_values('feature_importances',
                               ascending=False,
                               inplace=True)

    # plt.figure(figsize=(fig_size_x, fig_size_y))
    df_importances.plot.bar(x='features', y='feature_importances',
                            yerr='std_feature_importances', legend=None,
                            figsize=(fig_size_x, fig_size_y))
    plt.ylabel('Probabilidade')
