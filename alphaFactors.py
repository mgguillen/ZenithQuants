import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_ta as ta
from datetime import datetime
import requests
import nolds
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf


class AlphaFactors:

    def __init__(self, data):
        """
        Inicializa la clase con un DataFrame que contiene los datos necesarios para el cálculo de los factores.
        :param data: DataFrame de pandas.
        """
        self.data = data

    def momentum(self, window=20):
        """
        Calcula el factor de momentum para el DataFrame proporcionado.
        :param window: Número de días para calcular el momentum.
        :return: Serie de pandas con el momentum calculado.
        """
        self.etf_data[etf]['SMA'] = self.etf_data[etf].ta.sma(close='Close')
        self.etf_data[etf]["SMA6"] = self.etf_data[etf]["Close"].rolling(6).mean().shift(1)
        self.etf_data[etf]["SMA12"] = self.etf_data[etf]["Close"].rolling(12).mean().shift(1)
        self.etf_data[etf]["SMA24"] = self.etf_data[etf]["Close"].rolling(24).mean().shift(1)
        self.etf_data[etf]["IMP6"] = self.etf_data[etf]["SMA12"] - self.etf_data[etf]["SMA6"]
        self.etf_data[etf]["IMP18"] = self.etf_data[etf]["SMA24"] - self.etf_data[etf]["SMA6"]
        self.etf_data[etf]["IMP12"] = self.etf_data[etf]["SMA24"] - self.etf_data[etf]["SMA12"]
        mean6 = self.etf_data[etf]["IMP6"].mean()
        std6 = self.etf_data[etf]["IMP6"].std()
        mean18 = self.etf_data[etf]["IMP18"].mean()
        std18 = self.etf_data[etf]["IMP18"].std()
        mean12 = self.etf_data[etf]["IMP12"].mean()
        std12 = self.etf_data[etf]["IMP12"].std()
        self.etf_data[etf]["IMP6"] = (self.etf_data[etf]["IMP6"] - mean6) / std6
        self.etf_data[etf]["IMP18"] = (self.etf_data[etf]["IMP18"] - mean18) / std18
        self.etf_data[etf]["IMP12"] = (self.etf_data[etf]["IMP12"] - mean12) / std12

    def mean_reversion(self, window=20):
        """
        Calcula el factor de mean reversion.
        :param window: Número de días para calcular la media.
        :return: Serie de pandas con el mean reversion calculado.
        """
        historical_mean = self.data['Close'].rolling(window=window).mean()
        return (self.data['Close'] - historical_mean) / historical_mean

    # Añade más métodos para calcular otros factores alfa...
