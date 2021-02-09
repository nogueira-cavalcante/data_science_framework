"""Análise de correlação entre variáveis contínuas."""


def correlation_continuous_target_continuous_features(df, target_col,
                                                      features_cols=None,
                                                      corr_method='spearman',
                                                      fig_size_x=5,
                                                      fig_size_y=5):
    """
    Análise de correlação para target contínuo e features contínuas.

    Esta função utiliza testes de correlação linear para a análise de
    correlação.

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
    corr_method : string, optional
        Método de correlação ('pearson', 'kendall' ou 'spearman').
        O valor padrão é 'spearman'.
    fig_size_x : integer, optional
        Tamanho do eixo x para o gráfico. O valor padrão é 5.
    fig_size_y : integer, optional
        Tamanho do eixo y para o gráfico. O valor padrão é 5.

    Returns
    -------
    Gráfico mostrando a análise de correlação absoluta entre as variáveis
    features com o target, através do teste estatísticos lineare informado.

    """
    import matplotlib.pyplot as plt

    print('Índices próximos de zero: correlação linear fraca ou inexistente.')
    print('Índices próximos de um: correlação linear forte.')

    if target_col is None:
        target_col = list(df.columns).remove(target_col)

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
