"""Análise de tipos de variáveis."""

# -*- coding: utf-8 -*-


def type_of_features(df):
    """
    Verificação dos tipos de variáveis no DataFrame de entrada.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo os dados para a análise.

    Returns
    -------
    tabela : pandas DataFrame
        DataFrame contendo informações de análises sobre o DataFrame de
        entrada.
    lista_cat : list
        Lista contendo os nomes das colunas categóricas.
    lista_num : list
        Lista contendo os nomes das colunas contínuas.

    """
    import pandas as pd

    lista = []
    for i in range(len(df.columns)):
        coluna = df.iloc[:, i].name
        tipo = df.iloc[:, i].dtypes
        lista.append([coluna, tipo])

    tabela = pd.DataFrame(lista, columns=['coluna', 'dtype'])
    tab_cat = tabela[(tabela.dtype == 'object')]
    tab_num = tabela[(tabela.dtype == 'int64') |
                     (tabela.dtype == 'float64')]

    lista_cat = list(tab_cat.coluna)
    lista_num = list(tab_num.coluna)

    return tabela, lista_cat, lista_num
