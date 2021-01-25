"""
Funções de geração de gráficos para pré-investigação.
"""


def matriz_dispersao(df, target_col, features_cols,
                     fig_size_x=15, fig_size_y=15):

    """
    Cria uma matriz de gráficos, onde cada linha e cada coluna correspondem a
    uma feature ou target do DataFrame de entrada. Portanto, cada elemento
    da matriz corresponde a um plot linha X coluna. Caso a feature (ou target)
    da linha seja igual a da coluna um histograma (kde) será gerado. Esta
    função pode ser usada quando as variáveis envolvidas são contínuas.

    Args:
        df (pandas DataFrame): DataFrame contendo as features e o target.
        target_col (string): Nome do target.
        features_cols (list): Lista contendo os nomes das features.

    Imports:
        from pandas.plotting import scatter_matrix
    """

    from pandas.plotting import scatter_matrix

    df_temp = df[[target_col] + features_cols].copy()
    df_temp.rename(columns={target_col: target_col+' (target)'}, inplace=True)

    scatter_matrix(frame=df_temp, alpha=0.1,
                   figsize=(fig_size_x, fig_size_y), diagonal='kde')


def histogramas(df, target_col, features_cols, bins=20,
                fig_size_x=15, fig_size_y=15):

    """
    Geração de histogramas por coluna do DataFrame.

    Args:
        df (pandas DataFrame): DataFrame contendo as features e o target.
        target_col (string): Nome do target.
        features_cols (list): Lista contendo os nomes das features.
        bins=20 (integer): Quantidade de bins por histograma (valor padrão 20).

    """

    df_temp = df[[target_col] + features_cols].copy()
    df_temp.rename(columns={target_col: target_col+' (target)'}, inplace=True)

    df_temp.hist(bins=bins, figsize=(fig_size_x, fig_size_y))


def mapa_calor(df, target_col, features_cols, corr_method='pearson',
               fig_size_x=15, fig_size_y=15):

    """
    Geração de um mapa de calor do tipo coolwarm da correlação das variáveis
    features com o target. Esta função pode ser usada se as variáveis
    envolvidas são contínuas.

    Args:
        df (pandas DataFrame): DataFrame contendo as features e o target.
        target_col (string): Nome do target.
        features_cols (list): Lista contendo os nomes das features.
        corr_method='pearson' (string): Método de correlação ('pearson',
                                       'kendall' ou 'spearman')

    Imports:
        import matplotlib.pyplot as plt
        import seaborn as sns

    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    df_temp = df[[target_col] + features_cols].copy()
    df_temp.rename(columns={target_col: target_col+' (target)'}, inplace=True)

    corr = df_temp.corr(method=corr_method)

    plt.figure(figsize=(fig_size_x, fig_size_y))
    heatmap = sns.heatmap(round(corr, 2), annot=True, vmin=-1, vmax=1,
                          cmap="coolwarm", fmt='.2f', linewidths=.05)
    heatmap.set_title('Mapa de calor de correlação ' + corr_method,
                      fontdict={'fontsize': 15}, pad=12)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90,
                            fontsize=12)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0,
                            fontsize=12)


def correlacao_target_continuo_features_continuas(df, target_col,
                                                  features_cols,
                                                  corr_method='spearman',
                                                  fig_size_x=5,
                                                  fig_size_y=5):

    """
    Esta função só pode ser usada quando as varíveis target e features são
    contínuas. Geração de gráficos em barras mostrando o coeficiente (absoluto)
    de correlação das variáveis features com o target. As barras são
    mostradas em ordem decrescente.

    Args:
        df (pandas DataFrame): DataFrame contendo as features e o target.
        target_col (string): Nome do target.
        features_cols (list): Lista contendo os nomes das features.
        corr_method='pearson' (string): Método de correlação ('pearson',
                                       'kendall' ou 'spearman')

    Imports:
        import matplotlib.pyplot as plt

    """

    import matplotlib.pyplot as plt

    print('Índices próximos de zero: correlação linear fraca ou inexistente.')
    print('Índices próximos de um: correlação linear forte.')

    df_temp = df[[target_col] + features_cols].copy()

    corr = df_temp.corr(method=corr_method)
    corr = corr.loc[target_col, :].to_frame()
    corr.drop(target_col, axis=0, inplace=True)
    corr = corr.reset_index()
    corr.rename(columns={'index': 'features'}, inplace=True)
    corr[target_col] = corr[target_col].abs()
    corr.sort_values(target_col, ascending=False, inplace=True)

    corr.plot.bar(x='features', y=target_col, figsize=(fig_size_x,
                                                       fig_size_y))

    plt.ylabel('Índice absoluto', size=12)
    title_bar_plot = 'Análise de correlação ' + corr_method + \
                     ' do target com as features'
    plt.title(title_bar_plot, size=12)


def correlacao_target_continuo_features_categoricas(df, target_col,
                                                    features_cols,
                                                    plot_log10_anova=False,
                                                    fig_size_x=15,
                                                    fig_size_y=5):

    """
    Esta função só pode ser usada quando o target é contínuo e as features são
    categóricas.
    """

    from scipy import stats
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    print('######################################################')
    print('Teste estatístico ANOVA.')
    print('Hipótese nula: Não há diferenças entre os grupos.')
    print('Hipótese alternativa: Há diferenças entre os grupos.')
    print('\n')
    print('Se p_value menor que 0.05: rejeitar hipótese nula.')
    print('Se p_value maior que 0.05: não rejeitar hipótese nula.')
    print('\n')
    print('Quanto maior o teste anova maior a correlação.')
    print('######################################################')

    print('\n')
    print('\n')

    df_temp = df[[target_col] + features_cols].copy()

    features = []
    anova_test = []
    p_value = []

    for i in df_temp.drop(columns=[target_col]).columns:
        print('Feature:', i)
        list_target_group = df_temp.groupby(i)[target_col].apply(list)
        F, p = stats.f_oneway(*list_target_group)
        features.append(i)
        anova_test.append(F)
        p_value.append(p)
        print('Anova test:', F)
        print('p_value:', p)
        print('\n')

    df_anova = pd.DataFrame(np.array([features,
                                      anova_test,
                                      p_value]).transpose(),
                            columns=['features', 'anova', 'p_value'])
    df_anova['anova'] = df_anova['anova'].astype(float)
    df_anova['p_value'] = df_anova['p_value'].astype(float)
    df_anova.sort_values('anova', ascending=False, inplace=True)
    df_anova.loc[:, 'log10_anova'] = np.log10(df_anova.loc[:, 'anova'])

    fig, ax = plt.subplots(1, 2, figsize=(fig_size_x, fig_size_y))

    if plot_log10_anova:
        df_anova.plot.bar(x='features', y='log10_anova', ax=ax[0],
                          legend=None)
        ax[0].set_ylabel('log10 (anova test)')
    else:
        df_anova.plot.bar(x='features', y='anova', ax=ax[0],
                          legend=None)
        ax[0].set_ylabel('anova test')
    df_anova.plot.bar(x='features', y='p_value', ax=ax[1],
                      legend=None)
    ax[0].set_title('Target: ' + target_col)
    ax[1].set_title('Target: ' + target_col)
    ax[1].set_ylabel('p_value (anova)')


def correlacao_target_categorico_features_categoricas(df, target_col,
                                                      features_cols,
                                                      plot_log10_chi2=False,
                                                      fig_size_x=15,
                                                      fig_size_y=5):

    """
    Esta função só pode ser usada quando o target é categórico e as features
    são categóricas.
    """

    print('######################################################')
    print('Teste estatístico chi-quadrado.')
    print('Hipótese nula: Variáveis não são associadas.')
    print('Hipótese alternativa: Variáveis são associadas.')
    print('\n')
    print('Se p_value menor que 0.05: rejeitar hipótese nula.')
    print('Se p_value maior que 0.05: não rejeitar hipótese nula.')
    print('\n')
    print('Quanto maior o teste chi2 maior a associação.')
    print('######################################################')
    print('\n')
    print('\n')

    from scipy.stats import chi2_contingency
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df_temp = df[[target_col] + features_cols].copy()

    features = []
    chi2_test = []
    p_value = []

    for feature in df_temp.drop(columns=[target_col]).columns:
        print('Feature:', feature)
        contingency = pd.DataFrame(columns=df_temp[feature].unique(),
                                   index=df_temp[target_col].unique())
        for row in df_temp[target_col].unique():
            for column in df_temp[feature].unique():
                contingency_value = ((df_temp[target_col] == row) &
                                     (df_temp[feature] == column)).sum()
                contingency.loc[row, column] = contingency_value

        chi2, p, dof, expected = chi2_contingency(contingency.values)
        features.append(feature)
        chi2_test.append(chi2)
        p_value.append(p)
        print('chi2 test:', chi2)
        print('p_value:', p)
        print('\n')

    df_chi2 = pd.DataFrame(np.array([features,
                                     chi2_test,
                                     p_value]).transpose(),
                           columns=['features', 'chi2', 'p_value'])
    df_chi2['chi2'] = df_chi2['chi2'].astype(float)
    df_chi2['p_value'] = df_chi2['p_value'].astype(float)
    df_chi2.sort_values('chi2', ascending=False, inplace=True)
    df_chi2.loc[:, 'log10_chi2'] = np.log10(df_chi2.loc[:, 'chi2'])

    fig, ax = plt.subplots(1, 2, figsize=(fig_size_x, fig_size_y))

    if plot_log10_chi2:
        df_chi2.plot.bar(x='features', y='log10_chi2', ax=ax[0],
                         legend=None)
        ax[0].set_ylabel('log10 (chi2 test)')
    else:
        df_chi2.plot.bar(x='features', y='chi2', ax=ax[0],
                         legend=None)
        ax[0].set_ylabel('chi2 test')
    df_chi2.plot.bar(x='features', y='p_value', ax=ax[1],
                     legend=None)
    ax[0].set_title('Target: ' + target_col)
    ax[1].set_title('Target: ' + target_col)
    ax[1].set_ylabel('p_value (chi2)')


def correlacao_target_categorico_features_continuas(df, target_col,
                                                    features_cols,
                                                    fig_size_x=10,
                                                    fig_size_y=5):

    """
    Esta função só pode ser usada quando o target é categórico e as features
    são contínuas.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from scipy.stats import boxcox
    from sklearn.ensemble import ExtraTreesClassifier

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
