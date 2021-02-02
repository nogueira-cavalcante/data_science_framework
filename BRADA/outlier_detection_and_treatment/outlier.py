def outlier_detection(df, method='COPOD', columns=[], kwargs={}):
    """
    Função de identificação de outliers, baseado no pacote PyOD
    (https://pyod.readthedocs.io/en/latest/index.html).

    Args:
        df (pandas DataFrame): DataFrame contendo as colunas onde serão
                               identificados os outliers.
        method (string): Nome do método a ser usado para identificar os
                         outliers. Os métodos implementados nesta função são:
                         - 'ABOD': Angle-Based Outlier Detection;
                         - 'AutoEncoder': Fully connected AutoEncoder;
                         - 'CBLOF': Clustering-Based Local Outlier Factor;
                         - 'COF': Connectivity-Based Outlier Factor;
                         - 'COPOD': Copula-Based Outlier Detection;
                         - 'FeatureBagging': Feature Bagging;
                         - 'HBOS': Histogram-based Outlier Score;
                         - 'IForest': Isolation Forest;
                         - 'kNN': k Nearest Neighbors;
                         - 'LMDD': Deviation-based Outlier Detection;
                         - 'LODA': Lightweight On-line Detector of Anomalies;
                         - 'LOF': Local Outlier Factor;
                         - 'LOCI': Fast outlier detection using the local
                                   correlation integral;
                         - 'LSCP': Locally Selective Combination of Parallel
                                   Outlier Ensembles;
                         - 'MAD': 'Median Absolute Deviation;
                         - 'MCD': Minimum Covariance Determinant;
                         - 'MO_GAAL': Multiple-Objective Generative
                                      Adversarial Active Learning;
                         - 'OCSVM': One-Class Support Vector Machines;
                         - 'PCA': Principal Component Analysis;
                         - 'SOD': 'Subspace Outlier Detection;
                         - 'SO_GAAL': Single-Objective Generative Adversarial
                                      Active Learning;
                         - 'SOS': Stochastic Outlier Selection;
                         - 'VAE': Variational AutoEncoder;
                         - 'XGBOD': Extreme Boosting Based Outlier Detection
                                    (Supervised).
        columns (list): Lista de strings com os nomes das colunas para a
                        análise de detecção de outliers. Caso esse parâmetro
                        não seja informado a análise de detecção de outliers
                        será realizada em todas as colunas do DataFrame.
        kwargs (dict): Dicionário opcional contendo os parâmetros do método
                       de detecção de outliers escolhido. A relação de
                       parâmetros de cada método implementado é obtido em
                       https://pyod.readthedocs.io/en/latest/pyod.models.html#.
                       Por padrão esse parâmetro recebe um dicionário vazio,
                       fazendo com que os parâmetros do método escolhido
                       recebam seus valores padrões.
    Returns:
        pandas DataFrame contendo todas as colunas do DataFrame de entrada,
        além da coluna n_columns_outlier_flag (0 para normal e 1 para outlier).

    """

    print('Detecção de outliers...')

    if method == 'ABOD':
        from pyod.models.abod import ABOD
        print('Probabilistic method: ' +
              'Angle-Based Outlier Detection.')
        clf = ABOD(**kwargs)

    elif method == 'AutoEncoder':
        from pyod.models.auto_encoder import AutoEncoder
        print('Neural Networks method: ' +
              'Fully connected AutoEncoder.')
        clf = AutoEncoder(**kwargs)

    elif method == 'CBLOF':
        from pyod.models.cblof import CBLOF
        print('Proximity-Based method: ' +
              'Clustering-Based Local Outlier Factor.')
        clf = CBLOF(**kwargs)

    elif method == 'COF':
        from pyod.models.cof import COF
        print('Proximity-Based method: ' +
              'Connectivity-Based Outlier Factor.')
        clf = COF(**kwargs)

    elif method == 'COPOD':
        from pyod.models.copod import COPOD
        print('Probabilistic method: ' +
              'Copula-Based Outlier Detection.')
        clf = COPOD(**kwargs)

    elif method == 'FeatureBagging':
        from pyod.models.feature_bagging import FeatureBagging
        print('Outlier Ensembles method: ' +
              'Feature Bagging.')
        clf = FeatureBagging(**kwargs)

    elif method == 'HBOS':
        from pyod.models.hbos import HBOS
        print('Proximity-Based method: ' +
              'Histogram-based Outlier Score.')
        clf = HBOS(**kwargs)

    elif method == 'IForest':
        from pyod.models.iforest import IForest
        print('Outlier Ensembles method: ' +
              'Isolation Forest.')
        clf = IForest(**kwargs)

    elif method == 'kNN':
        from pyod.models.knn import KNN
        print('Proximity-Based method: ' +
              'k Nearest Neighbors.')
        clf = KNN(**kwargs)

    elif method == 'LMDD':
        from pyod.models.lmdd import LMDD
        print('Linear Model method: ' +
              'Deviation-based Outlier Detection.')
        clf = LMDD(**kwargs)

    elif method == 'LODA':
        from pyod.models.loda import LODA
        print('Outlier Ensembles method: ' +
              'Lightweight On-line Detector of Anomalies.')
        clf = LODA(**kwargs)

    elif method == 'LOF':
        from pyod.models.lof import LOF
        print('Proximity-Based method: ' +
              'Local Outlier Factor.')
        clf = LOF(**kwargs)

    elif method == 'LOCI':
        from pyod.models.loci import LOCI
        print('Proximity-Based method: ' +
              'Fast outlier detection using the local correlation integral.')
        clf = LOCI(**kwargs)

    elif method == 'LSCP':
        from pyod.models.lscp import LSCP
        print('Outlier Ensembles method: ' +
              'Locally Selective Combination of Parallel Outlier Ensembles.')
        clf = LSCP(**kwargs)

    elif method == 'MAD':
        from pyod.models.mad import MAD
        print('Probabilistic method: ' +
              'Median Absolute Deviation.')
        clf = MAD(**kwargs)

    elif method == 'MCD':
        from pyod.models.mcd import MCD
        print('Linear Model: ' +
              'Minimum Covariance Determinant.')
        clf = MCD(**kwargs)

    elif method == 'MO_GAAL':
        from pyod.models.mo_gaal import MO_GAAL
        print('Neural Networks method: ' +
              'Multiple-Objective Generative Adversarial Active Learning.')
        clf = MO_GAAL(**kwargs)

    elif method == 'OCSVM':
        from pyod.models.ocsvm import OCSVM
        print('Linear Model method: ' +
              'One-Class Support Vector Machines.')
        clf = OCSVM(**kwargs)

    elif method == 'PCA':
        from pyod.models.pca import PCA
        print('Linear Model method: ' +
              'Principal Component Analysis.')
        clf = PCA(**kwargs)

    elif method == 'SOD':
        from pyod.models.sod import SOD
        print('Proximity-Based method: ' +
              'Subspace Outlier Detection.')
        clf = SOD(**kwargs)

    elif method == 'SO_GAAL':
        from pyod.models.so_gaal import SO_GAAL
        print('Neural Networks method: ' +
              'Single-Objective Generative Adversarial Active Learning.')
        clf = SO_GAAL(**kwargs)

    elif method == 'SOS':
        from pyod.models.sos import SOS
        print('Probabilistic method: ' +
              'Stochastic Outlier Selection.')
        clf = SOS(**kwargs)

    elif method == 'VAE':
        from pyod.models.vae import VAE
        print('Neural Networks method: ' +
              'Variational AutoEncoder.')
        clf = VAE(**kwargs)

    elif method == 'XGBOD':
        from pyod.models.xgbod import XGBOD
        print('Outlier Ensembles method: ' +
              'Extreme Boosting Based Outlier Detection (Supervised).')
        clf = XGBOD(**kwargs)

    else:
        raise Exception('Método inválido ou não se encontra implementado.')

    df_temp = df.copy()

    if len(columns) == 0:
        columns = df_temp.columns

    for col in columns:
        outlier_column = col + '_' + 'outlier_flag'
        clf.fit(df_temp[[col]])
        outlier_flag = clf.predict(df_temp[[col]])
        df_temp.loc[:, outlier_column] = outlier_flag

    return df_temp


def outlier_treatment(df, columns=[], method='removal',
                      remove_outlier_column=True, constant_value=0):

    """
    Função de tratamento de outliers, sendo necessário ter usado a função
    outlier_detection antes.

    Args:
        df (pandas DataFrame): DataFrame contendo as colunas já identificadas
                               como outliers.
        method (string): Nome do método a ser usado para tratar os outliers.
                         Os métodos implementados nesta função são:
                         - 'removal': Remoção das linhas do outliers;
                         - 'constant': Subtituição do outlier pelo valor
                                       definido em constant_value;
                         - 'min': Subtituição do outlier pelo valor
                                  mínimo da coluna;
                         - 'max': Subtituição do outlier pelo valor
                                  máximo da coluna;
                         - 'mean': Subtituição do outlier pelo valor
                                   máédio da coluna;
                         - 'median': Subtituição do outlier pelo valor
                                     da medianda da coluna;
                         - 'mode': Subtituição do outlier pela moda
                                   da coluna;
                         - 'bfill': Subtituição do outlier pelo valor
                                    posterior ao outlier;
                         - 'ffill': Subtituição do outlier pelo valor
                                    anterior ao outlier;
                         - 'interpolate': Interpolação linear (somente
                                          para séries temporais onde o
                                          index é datetime).
        remove_outlier_column (boolean): Caso True, irá remover do DataFrame
                                         as colunas que identificam os
                                         outliers.
        constant_value (float): Valor constante para ser usado no método
                               'constant'.

    Returns:
        pandas DataFrame contendo todas as colunas do DataFrame de entrada
        com os outliers tratados, confome o método escolhido.

    """

    import numpy as np

    print('Tratamento de outliers...')
    print('Método: ' + method)

    df_temp = df.copy()

    if len(columns) == 0:
        outlier_columns = df_temp.filter(regex='_outlier_flag').columns

    else:
        outlier_columns = []
        for col in df_temp[columns].columns:
            outlier_columns.append(col + '_outlier_flag')

    print('Total de colunas a serem analisadas:',
          df_temp[outlier_columns].shape[1])

    for outlier_column in outlier_columns:
        col = outlier_column.replace('_outlier_flag', '')

        df_temp.loc[df_temp[outlier_column] == 1, col] = np.nan

        if method == 'removal':
            df_temp.dropna(subset=[col], inplace=True)

        elif method == 'constant':
            df_temp[col] = df_temp[col].fillna(constant_value)

        elif method == 'min':
            df_temp[col] = df_temp[col].fillna(df_temp[col].min())

        elif method == 'max':
            df_temp[col] = df_temp[col].fillna(df_temp[col].max())

        elif method == 'mean':
            df_temp[col] = df_temp[col].fillna(df_temp[col].mean())

        elif method == 'median':
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())

        elif method == 'mode':
            df_temp[col] = df_temp[col].fillna(df_temp[col].mode().iloc[0])

        elif method == 'bfill':
            df_temp[col] = df_temp[col].fillna(method='bfill')

        elif method == 'ffill':
            df_temp[col] = df_temp[col].fillna(method='ffill')

        elif method == 'interpolate':
            index_type = str(df_temp.index.dtype)
            if 'datetime' not in index_type:
                raise Exception('O indice do DataFrame deve estar no ' +
                                'formato datetime.')
            df_temp.sort_index(inplace=True)

            df_temp[col] = df_temp[col].interpolate(method='linear')

        else:
            raise Exception('Método inválido.')

        if remove_outlier_column:
            df_temp.drop(columns=[outlier_column], inplace=True)

    return df_temp
