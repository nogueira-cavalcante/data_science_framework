"""Relatório de errors das features."""


def report_error_features(modelos, X_val, y_val, lista_predicoes, target):
    """
    Geração de um gráfico line, através dos dados de validação X_val e y_val.

    Parameters
    ----------
    modelos : list
        Lista contendo objetos do tipo classifier ou regressor ou uma classe
        implementando ambos os métidos fit e predict, similares ao de API
        scikit-learn.
    X_val : pandas DataFrame
        Amostra de validação treino para a geração do gráfico.
    y_val : pandas DataFrame
        Amostra de target correspondente à X_val.
    lista_predicoes : list
        Lista de predições dos modelos listados em "modelos" (y_pred).
    target : string
        O nome da coluna do target.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_predicoes = pd.concat(lista_predicoes, axis=1)
    df_predicoes = pd.concat([df_predicoes, y_val.reset_index(drop=True)],
                             axis=1, join='inner')
    X_val_temp = X_val[modelos.top_features.iloc[0]].reset_index(drop=True)
    df_predicoes = pd.concat([X_val_temp, df_predicoes], axis=1)

    for i in list(modelos.tipo):
        pred = df_predicoes.apply(lambda x: abs(x[i] - x[target]), axis=1)
        df_predicoes[i+'_real_dif'] = pred

    plt.figure(figsize=(10, 10))
    plt.rcParams['font.size'] = 10
    ax1 = sns.scatterplot(data=df_predicoes, sizes=20, x=target,
                          y=modelos.tipo.iloc[0], alpha=0.5, s=15)
    print(int(df_predicoes[target].max()))
    ax1.plot([0, int(df_predicoes[target].max())],
             [0, int(df_predicoes[target].max())], '--')
    ax1.set_title('Valores reais e previsões')

    plt.figure(figsize=(10, 10))
    ax1 = sns.lineplot(data=df_predicoes, x=target,
                       y=modelos.tipo.iloc[0]+'_real_dif')
    ax1.set_title('Erros x faixa do target')

    return df_predicoes.sort_values(modelos.iloc[0, 0]+'_real_dif',
                                    ascending=False).head(10)
