"""Geração de mapa de matriz de dispersão."""


def dispersion_matrix(df, target_col, features_cols=None,
                      fig_size_x=15, fig_size_y=15):
    """
    Geração de matriz de dispersão.

    Cria uma matriz de gráficos, onde cada linha e cada coluna correspondem a
    uma feature ou target do DataFrame de entrada. Portanto, cada elemento
    da matriz corresponde a um plot linha X coluna. Caso a feature (ou target)
    da linha seja igual a da coluna um histograma (kde) será gerado. Esta
    função pode ser usada quando as variáveis envolvidas são contínuas.

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
        Tamanho do eixo y para o gráfico. O valor padrão é 15.

    Returns
    -------
    Gráfico de matriz de dispersão.

    """
    from pandas.plotting import scatter_matrix

    if target_col is None:
        target_col = list(df.columns).remove(target_col)

    df_temp = df[[target_col] + features_cols].copy()
    df_temp.rename(columns={target_col: target_col+' (target)'}, inplace=True)

    scatter_matrix(frame=df_temp, alpha=0.1,
                   figsize=(fig_size_x, fig_size_y), diagonal='kde')
