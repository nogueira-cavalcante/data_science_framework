"""Inicialização do pacote."""

from .quality_check import quality_check
from .train_test_split import train_test_split
from .data_cleaning import data_cleaning
from .feature_engineering import feature_engineering
from .exploratory_analysis import exploratory_analysis
from .feature_selection import feature_selection
from .automl import automl
from .error_analysis import error_analysis
from .result_interpretation import result_interpretation
from .time_series_analysis import time_series_analysis
# production pipeline
# monitoring

__version__ = '0.0.1'
__dist_name__ = 'Data Science Framework'
