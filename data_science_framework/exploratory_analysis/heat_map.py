"""Geração de mapa de calor."""


def heat_map(df, target_col, features_cols=None, corr_method='pearson',
             fig_size_x=15, fig_size_y=15):
    """
    Geração de mapa de calor do tipo coolwarm.

    Esta função pode ser usada se as variáveis envolvidas são contínuas.

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
        Método de correlação ('pearson', kendall' ou 'spearman').
        O valor padrão é 'pearson'.
    fig_size_x : integer, optional
        Tamanho do eixo x para o gráfico. O valor padrão é 15.
    fig_size_y : integer, optional
        Tamanho do eixo y para o gráfico. O valor padrão é 15.

    Returns
    -------
    Mapa de calor mostrando os indíces de correlação entre as variáveis de
    no DataFrame de entrada.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if target_col is None:
        target_col = list(df.columns).remove(target_col)

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
