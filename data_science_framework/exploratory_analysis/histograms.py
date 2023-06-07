"""Geração de histogramas."""


def histograms(df, target_col, features_cols=None, bins=20,
               fig_size_x=15, fig_size_y=15):
    """
    Geração de histogramas por coluna do DataFrame.

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
    bins : boolean, optional
        Quantidade de bins por histograma. O valor padrão é 20.
    fig_size_x : integer, optional
        Tamanho do eixo x para o gráfico. O valor padrão é 15.
    fig_size_y : integer, optional
        Tamanho do eixo y para o gráfico. O valor padrão é 15.

    Returns
    -------
    Gráficos mostrando as distribuições (histogramas) de cada coluna do
    DataFrame de entrada

    """
    if target_col is None:
        target_col = list(df.columns).remove(target_col)

    df_temp = df[[target_col] + features_cols].copy()
    df_temp.rename(columns={target_col: target_col+' (target)'}, inplace=True)

    df_temp.hist(bins=bins, figsize=(fig_size_x, fig_size_y))
