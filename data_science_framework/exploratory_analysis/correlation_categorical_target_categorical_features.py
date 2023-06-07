"""Análise de correlação entre variáveis categóricas."""


def correlation_categorical_target_categorical_features(df, target_col,
                                                        features_cols=None,
                                                        plot_log10_chi2=False,
                                                        fig_size_x=15,
                                                        fig_size_y=5):
    """
    Análise de correlação para target categórico e features categóricas.

    Esta função utiliza o teste chi-quadrado para a análise de correlação.

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
    plot_log10_chi2 : boolean, optional
        Caso True, o gráfico resultante estará em escala logarítmica.
        O valor padrão é False.
    fig_size_x : integer, optional
        Tamanho do eixo x para o gráfico. O valor padrão é 15.
    fig_size_y : integer, optional
        Tamanho do eixo y para o gráfico. O valor padrão é 5.

    Returns
    -------
    Gráfico mostrando a análise de correlação entre as variáveis features com
    o target, através do teste estatístico chi-quadrado.

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

    if target_col is None:
        target_col = list(df.columns).remove(target_col)

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
