"""
Funções de geração de gráficos para pré-investigação.
"""


def matriz_dispersao(df, target_col, features_cols):

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

    scatter_matrix(frame=df_temp, alpha=0.1,  figsize=(15, 15), diagonal='kde')


def histogramas(df, target_col, features_cols, bins=20):

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

    df_temp.hist(bins=bins, figsize=(15, 15))


def mapa_calor(df, target_col, features_cols, corr_method='pearson'):

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

    plt.figure(figsize=(15, 15))
    heatmap = sns.heatmap(round(corr, 2), annot=True, vmin=-1, vmax=1,
                          cmap="coolwarm", fmt='.2f', linewidths=.05)
    heatmap.set_title('Mapa de calor de correlação ' + corr_method,
                      fontdict={'fontsize': 15}, pad=12)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90,
                            fontsize=12)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0,
                            fontsize=12)


def grafico_barras_corr_variaveis_continuas(df, target_col, features_cols,
                                            corr_method='spearman'):

    """
    Geração de gráficos em barras mostrando o coeficiente (absoluto) de
    correlação das variáveis features com o target. As barras são
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

    df_temp = df[[target_col] + features_cols].copy()

    corr = df_temp.corr(method=corr_method)
    corr = corr.loc[target_col, :].to_frame()
    corr.drop(target_col, axis=0, inplace=True)
    corr = corr.reset_index()
    corr.rename(columns={'index': 'features'}, inplace=True)
    corr[target_col] = corr[target_col].abs()
    corr.sort_values(target_col, ascending=False, inplace=True)

    corr.plot.bar(x='features', y=target_col, figsize=(15, 10))

    plt.ylabel('Índice absoluto', size=12)
    title_bar_plot = 'Análise de correlação ' + corr_method + \
                     ' do target com as features'
    plt.title(title_bar_plot, size=12)
