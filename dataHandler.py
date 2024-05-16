import warnings
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pandas_market_calendars as mcal
warnings.filterwarnings("ignore")

class DataHandler:
    # Define valores predeterminados para ETFs y métricas económicas, y directorios para guardar los datos.
    DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]
    DEFAULT_METRICS = ["GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "BOPGSTB", "PPIACO", "UMCSENT", "T10Y2Y", 'TB3MS']
    SAVE = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/price'
    SAVE_PATH = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/daily'
    SAVE_PATH_SERIES = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/series'

    def __init__(self, etfs=None, metrics=None, start_date='2010-01-01', end_date=None, freq='M',
                 start_back='2020-01-02', end_back='2024-04-10', fetch_fred=True, technical=True):
        """
        Inicializamos para cargar datos históricos.
        :param etfs: Lista de ETFs a cargar.
        ######### ahi que quitarlo :param metrics: Métricas económicas para cargar desde FRED.
        :param start_date: Fecha de inicio para la carga de datos.
        :param end_date: Fecha final para la carga de datos.
        :param freq: Frecuencia de los datos (diaria, mensual, etc.).
        :param start_back: Fecha de inicio para el backtesting.
        :param end_back: Fecha final para el backtesting.
        ##########:param fetch_fred: Si se deben cargar datos de FRED.
        ##########:param technical: Si se deben calcular indicadores técnicos.
        """
        self.etfs = etfs if etfs else self.DEFAULT_ETFs
        self.metrics = metrics if metrics else self.DEFAULT_METRICS
        self.start_date = start_date
        self.adjusted_start_date = (pd.to_datetime(self.start_date) - DateOffset(months=28)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.freq = freq
        self.scalers = {}
        self.etf_data = {}
        self.start_back = start_back
        self.end_back = end_back
        start_back_datetime = datetime.strptime(self.start_back, '%Y-%m-%d')
        end_date_fred = start_back_datetime - timedelta(days=1)
        self.end_date_fred_str = end_date_fred.strftime('%Y-%m-%d')
        self.API_KEY_FRED = 'd6d360dc757c1441a1a73cf568e60e1c'
        self.fetch_fred = fetch_fred
        self.technical = technical

        self.data_dir = Path(self.SAVE_PATH_SERIES)
        self.data_dir.mkdir(exist_ok=True)  # Crea el directorio si no existe
        self.data_dir_daily = Path(self.SAVE_PATH)
        self.data_dir_daily.mkdir(exist_ok=True)  # Crea el directorio si no existe
        self.data_dir_price = Path(self.SAVE)
        self.data_dir_price.mkdir(exist_ok=True)  # Crea el directorio si no existe

    def load_data(self):
        """
        Cargamos los datos históricos para cada ETF, verificamos  si tenemos los datos almacenados antes
        de cargar desde yfinance y actualizamos los datos si es necesario.
        """
        lista_df = []
        for etf in self.DEFAULT_ETFs:
            file_path = self.data_dir_price / f"{etf}.csv"
            if file_path.exists():
                data = pd.read_csv(file_path)
                data['etf'] = etf
                # Verifica que 'date' es el tipo correcto después de cargar
                if not data.empty:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                    max_date = data.index.max()
                    new_start_date = max_date + pd.DateOffset(days=1)
                    if new_start_date <= pd.to_datetime(self.end_date_fred_str):
                        # Actualiza los datos desde yfinance si son necesarios
                        new_data = yf.Ticker(etf).history(start=new_start_date.strftime('%Y-%m-%d'), end=self.end_date)
                        new_data.index = new_data.index.tz_localize(None)
                        new_data = new_data.rename(columns=str.lower).rename(columns={'dividends': 'dividend', 'stock splits': 'split', 'capital gains': 'gains'}).fillna(method='ffill').fillna(method='bfill')
                        new_data['etf'] = etf
                        data = pd.concat([data, new_data])
                        data.to_csv(file_path, index=True)  # Guardar la data actualizada
            else:
                # Carga los datos desde yfinance si no existe archivo local
                data = yf.Ticker(etf).history(start=self.adjusted_start_date, end=self.end_date)
                data.index = data.index.tz_localize(None)
                data = data.rename(columns=str.lower).rename(columns={'dividends': 'dividend', 'stock splits': 'split', 'capital gains': 'gains'}).fillna(method='ffill').fillna(method='bfill')
                data['etf'] = etf
                data.to_csv(file_path, index=True)
            data = data.resample(self.freq).last()  # Resamplea los datos a la frecuencia especificada
            data = data.reset_index()
            lista_df.append(data)

        data_ohlcv = pd.concat(lista_df, ignore_index=True)  # Combina todos los DataFrames en uno solo
        data_ohlcv = data_ohlcv[data_ohlcv["date"]<=self.end_date_fred_str]
        #print("data handler",data_ohlcv.tail())
        return data_ohlcv

    def preprocess_data(self, data=None):
        if data is None:
            print("Por favor introduzca un Data Frame")
            return
        else:
            columnas_numericas = data.select_dtypes(include=['number']).columns
            sc = StandardScaler()
            data[columnas_numericas] = sc.fit_transform(data[columnas_numericas])
            self.scalers['data_scaler'] = sc  # Almacenar el scaler para uso futuro
            return data

    def reverse_preprocess_data(self, data=None):
        if data is None:
            print("Por favor introduzca un Data Frame")
            return
        elif 'data_scaler' not in self.scalers:
            print("No se ha escalado ningún dato previamente.")
            return
        else:
            columnas_numericas = data.select_dtypes(include=['number']).columns
            sc = self.scalers['data_scaler']  # Recuperar el scaler almacenado
            data[columnas_numericas] = sc.inverse_transform(data[columnas_numericas])
            return data

    def load_data_portfolio(self, etfs = None):
        """
        Cargamos los datos de precios de los ETFs desde los archivos locales para la optimización del portfolio.
        :return: DataFrame con los precios de cierre ajustados de los ETFs seleccionados.
        """
        if etfs is None or len(etfs) == 0:
            n_etfs = [col for col in self.etfs if col != "SPY"]
        else:
            n_etfs = etfs

        # Inicializar un DataFrame vacío para los datos del portfolio
        data_portfolio = pd.DataFrame()

        # Cargar los datos desde archivos locales
        for etf in n_etfs:
            file_path = self.data_dir_price / f"{etf}.csv"
            if file_path.exists():
                data_etf = pd.read_csv(file_path, index_col='date', parse_dates=True)
                data_portfolio[etf] = data_etf['close']

        return data_portfolio


