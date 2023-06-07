"""Análise de correlação entre variáveis categóricas e contínuas."""


def correlation_continuous_target_categorical_features(df, target_col,
                                                       features_cols=None,
                                                       plot_log10_anova=False,
                                                       fig_size_x=15,
                                                       fig_size_y=5):
    """
    Análise de correlação para target contínuo e features categóricas.

    Esta função utiliza o teste estatístico ANOVA para a análise de correlação.

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
    plot_log10_anova : boolean, optional
        Caso True, o gráfico resultante estará em escala logarítmica.
        O valor padrão é False.
    fig_size_x : integer, optional
        Tamanho do eixo x para o gráfico. O valor padrão é 15.
    fig_size_y : integer, optional
        Tamanho do eixo y para o gráfico. O valor padrão é 5.

    Returns
    -------
    Gráfico mostrando a análise de correlação entre as variáveis features com
    o target, através do teste estatístico ANOVA.

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

    if target_col is None:
        target_col = list(df.columns).remove(target_col)

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
