"""Pré-investigação dos dados."""


def null_analysis(df):
    """
    Verificação de nulos no DataFrame de entrada.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo os dados para a análise.

    Returns
    -------
    tabela : pandas DataFrame
        DataFrame contendo informações de análises sobre o DataFrame de
        entrada.
    lista_cat_nulos : list
        Lista contendo os nomes das colunas categóricas que contêm nulos.
    lista_num_nulos : list
        Lista contendo os nomes das colunas contínuas que contêm nulos.

    """
    import pandas as pd

    lista = []

    for i in range(len(df.columns)):
        coluna = df.iloc[:, i].name
        tipo = df.iloc[:, i].dtypes
        nulos = df.iloc[:, i].isnull().sum()
        percent_nulos = (df.iloc[:, i].isnull().sum())/len(df)
        lista.append([coluna, tipo, nulos, percent_nulos])

    tabela = pd.DataFrame(lista, columns=['coluna',
                                          'dtype',
                                          'nulos',
                                          'percent_nulos'])
    tab_cat = tabela[(tabela.dtype == 'object') & (tabela.nulos > 0)]
    tab_num = tabela[((tabela.dtype == 'int64') |
                      (tabela.dtype == 'float64')) &
                     (tabela.nulos > 0)]

    lista_cat_nulos = list(tab_cat.coluna)
    lista_num_nulos = list(tab_num.coluna)

    return tabela, lista_cat_nulos, lista_num_nulos


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
