# -*- coding: utf-8 -*-
"""
Análise de entropia para séries temporais.
"""

def entropy(x, analysis_kind='permutation', order=3, delay=1, normalize=False,
            sf=10, method='fft', nperseg=None, axis=-1):
    """
    Análise de entropia para séries temporais.
    
    Esta função utiliza o pacote Antropy. Para mais detalhes veja em
    https://github.com/raphaelvallat/antropy.

    Parameters
    ----------
    x : pandas Series, list or np.array
        Série temporal unidimensional ordenada.
    analysis_kind : string, opcional
        Tipo de análise de entropia:
            - 'permutation';
            - 'spectral';
            - 'singular_value_decomposition'.
        O valor padrão é 'permutation'.
    order : integer, opcional
        Ordem da entropy (usado para os modos 'permutation' e 
        'singular_value_decomposition'). O valor padrão é 3.
    delay : integer, optional
        Delay de tempo (usado para os modos 'permutation' e 
        'singular_value_decomposition'). O valor padrão é 1.
    normalize : boolean, optional
        Se True, divide por log2(order!) para normalizar a entropia entre 0 e
        1. Caso contrário, retorna a entropia em bit (usado em todos os modos).
        O valor padrão é False.
    sf : float, optional
        Frequência de amostragem, em Hz (usado apenas no modo 'spectral').
        O valor padrão é 10.
    method : string, optional
        Método para estimação espectral (usado apenas no modo 'spectral'):
            - 'fft': Transformada de Fourier (scipy.signal.periodogram);
            - 'welch': Periodograma de Welch (scipy.signal.welch).
        O valor padrão é 'fft'.
    nperseg : integer or None, opcional
        Tamanho de cada ssegmento de FFT para o método Welch
        (usado apenas no modo 'spectral'). O valor padrão é None.
    axis : integer, opcional
        Eixo ao longo o qual a entropia é calculada (usado apenas no modo
        'spectral'). O valor padrão é -1 (último).

    Returns
    -------
    entropy : float
        Valor calculado para a entropia.

    """
    import antropy as ant
    
    if analysis_kind == 'permutation':
        entropy = ant.perm_entropy(x=x,
                                   order=order,
                                   delay=delay,
                                   normalize=normalize)
        
    elif analysis_kind == 'spectral':
        entropy = ant.spectral_entropy(x=x,
                                       sf=sf,
                                       method=method,
                                       nperseg=nperseg,
                                       normalize=normalize,
                                       axis=axis)
        
    elif analysis_kind == 'singular_value_decomposition':
        entropy = ant.svd_entropy(x=x,
                                  order=order,
                                  delay=delay,
                                  normalize=normalize)
    
    else:
        raise Exception('O valor em analysis_kind é inválido. Veja a ' + 
                        'documentação da função para rever os possíveis ' + 
                        'valores para o parâmetro.')
        
    return entropy
