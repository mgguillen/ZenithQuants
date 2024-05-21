import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_ta as ta
from datetime import datetime
import requests
import nolds
from scipy.stats import skew, kurtosis
import warnings
import yfinance as yf
from scipy.stats import skew, kurtosis
import fbm
from hurst import compute_Hc
from pathlib import Path

warnings.filterwarnings("ignore")

class AlphaFactors:
    DEFAULT_METRICS = ["GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "BOPGSTB", "PPIACO", "UMCSENT", "T10Y2Y", 'TB3MS']
    SAVE = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/price'
    SAVE_PATH = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/daily'
    SAVE_PATH_SERIES = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/series'

    def __init__(self, data, end_date_fred_str='2020-01-02', start_date=None, rend=1, benchmark=None):
        """
        Inicializamos AlphaFactors.
        :param data: DataFrame con los datos de precios.
        :param end_date_fred_str: Esta fecha nos indica cuando inicia el backtesting en cada ciclo.
        :param start_date: Fecha de inicio para la la carga de datos.
        """

        self.data = data
        self.rend = rend
        self.benchmark = benchmark
        self.data['close_or'] = self.data['close']
        self.data['close'] = np.log(self.data['close'] / self.data['close'].shift(1))
        if self.rend == 1 and benchmark is not None:
            # Resetear índices
            spy_df_reset = benchmark.reset_index(drop=True).fillna(method='bfill').fillna(method='ffill')
            etf_df_reset = self.data['close'].reset_index(drop=True).fillna(method='bfill').fillna(method='ffill')
            if len(etf_df_reset) == len(spy_df_reset):
                difference = np.subtract(etf_df_reset.to_numpy(), spy_df_reset.to_numpy())
                self.data['close'] = pd.Series(difference, index=self.data['close'].index)
            else:
                print("Error: Las series de datos no están alineadas en longitud.")
        self.API_KEY_FRED = 'd6d360dc757c1441a1a73cf568e60e1c'
        self.end_date_fred_str = end_date_fred_str
        self.start_date = start_date
        self.data_dir = Path(self.SAVE_PATH_SERIES)
        self.data_dir.mkdir(exist_ok=True)  # Crea el directorio si no existe
        self.data_dir_daily = Path(self.SAVE_PATH)
        self.data_dir_daily.mkdir(exist_ok=True)  # Crea el directorio si no existe
        self.data_dir_price = Path(self.SAVE)
        self.data_dir_price.mkdir(exist_ok=True)  # Crea el directorio si no existe

    def add_time(self):
        """
        Añadimos columnas de mes y semana del año.
        """
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['month'] = self.data['date'].dt.month  # Extrae el mes de la fecha
        self.data['week_of_year'] = self.data['date'].dt.isocalendar().week.astype(int)

    def add_historical_volatility(self, windows=[3, 6, 12, 24, 48]):
        """
        Añadimos la volatilidad histórica calculada como desviación estándar de los retornos.
        :param windows: Lista de ventanas de tiempo para calcular la volatilidad.
        """

        for window in windows:
            self.data[f'volatility_{window}m'] = self.data['close'].rolling(window).std() * np.sqrt(12)

    def add_skewness_kurtosis(self, windows=[3, 6, 12, 24, 48]):
        """
        Añade skewness y kurtosis calculadas sobre ventanas de tiempo especificadas.
        :param windows: Lista de ventanas de tiempo para calcular los indicadores.
        """
        for window in windows:
            self.data[f'skewness_{window}m'] = self.data['close'].rolling(window).apply(skew, raw=True)
            self.data[f'kurtosis_{window}m'] = self.data['close'].rolling(window).apply(kurtosis, raw=True)

    def add_momentum_indicators(self, windows=[3, 6, 12, 24, 48]):
        """
        Calculamos varios indicadores de momento utilizando la biblioteca pandas_ta.
        :param windows: Lista de ventanas de tiempo para calcular los indicadores.
        """
        for window in windows:
            self.data[f'RSI_{window}m'] = self.data.ta.rsi(close='close', length=window)
            self.data[f'MACD_{window}m'] = self.data.ta.macd(close='close', length=window)['MACD_12_26_9']
            self.data[f'PPO_{window}m'] = self.data.ta.ppo(close='close', length=window)['PPO_12_26_9']
            self.data[f'ROC_{window}m'] = self.data.ta.roc(close='close', length=window)
            self.data[f'ADX_{window}m'] = self.data.ta.adx(high='high', low='low', close='close', length=window)[f'ADX_{window}']

        for i in range(len(windows) - 1):
            short_window = windows[i]
            long_window = windows[i + 1]
            short_key_RSI = f'RSI_{short_window}m'
            long_key_RSI = f'RSI_{long_window}m'
            short_key_MACD = f'MACD_{short_window}m'
            long_key_MACD = f'MACD_{long_window}m'
            short_key_PPO = f'PPO_{short_window}m'
            long_key_PPO = f'PPO_{long_window}m'
            short_key_ROC = f'ROC_{short_window}m'
            long_key_ROC = f'ROC_{long_window}m'
            short_key_ADX = f'ADX_{short_window}m'
            long_key_ADX = f'ADX_{long_window}m'

            diff_key_PPO = f'momentum_diff_PPO_{short_window}m_{long_window}m'
            diff_key_RSI = f'momentum_diff_RSI_{short_window}m_{long_window}m'
            diff_key_MACD = f'momentum_diff_MACD_{short_window}m_{long_window}m'
            diff_key_ROC = f'momentum_diff_ROC_{short_window}m_{long_window}m'
            diff_key_ADX = f'momentum_diff_ADX_{short_window}m_{long_window}m'

            # Ahora realiza la resta, asegurándose de que ambos campos están libres de NaN
            self.data[diff_key_PPO] = self.data[short_key_PPO] - self.data[long_key_PPO]
            self.data[diff_key_RSI] = self.data[short_key_RSI] - self.data[long_key_RSI]
            self.data[diff_key_MACD] = self.data[short_key_MACD] - self.data[long_key_MACD]
            self.data[diff_key_ROC] = self.data[short_key_ROC] - self.data[long_key_ROC]
            self.data[diff_key_ADX] = self.data[short_key_ADX] - self.data[long_key_ADX]


    def add_bollinger_bands(self, windows=[3, 6, 12, 24, 48], num_std=2):
        """
        Calculamos las Bandas de Bollinger para diferentes ventanas de tiempo.
        :param windows: Lista de ventanas de tiempo para calcular los indicadores.
        :param num_std: Número de desviaciones estándar para las bandas.
        """

        for window in windows:
            rolling_mean = self.data['close'].rolling(window).mean()
            rolling_std = self.data['close'].rolling(window).std()
            self.data[f'bollinger_high_{window}'] = rolling_mean + (rolling_std * num_std)
            self.data[f'bollinger_low_{window}'] = rolling_mean - (rolling_std * num_std)

    def fetch_fred_data(self, serie):
        """
        Descargamos datos económicos de la API de FRED utilizando un identificador de serie.
        :param serie: Identificador de la serie a descargar.
        :return: DataFrame con la serie descargada.
        """

        url = (
            f"https://api.stlouisfed.org/fred/series/observations?"
            f"series_id={serie}&api_key={self.API_KEY_FRED}"
            f"&file_type=json&observation_start={self.start_date}"
            f"&observation_end={self.end_date_fred_str}"
        )

        response = requests.get(url).json()
        try:
            df_fred = pd.DataFrame(response['observations'])
            df_fred['date'] = pd.to_datetime(df_fred['date'])
            df_fred['date'] = df_fred['date'].dt.tz_localize(None)
            df_fred[serie] = pd.to_numeric(df_fred['value'], errors='coerce')
            return df_fred[['date', serie]]
        except KeyError:
            print(url)
            print(f"No se encontraron datos para la serie {serie}.")
            return pd.DataFrame()

    def add_fetch_fred(self):
        """
        Añadmos los datos económicos descargados o recuperados de FRED al DataFrame principal.
        """
        for serie in self.DEFAULT_METRICS:
            file_path = self.data_dir / f"{serie}.csv"
            if file_path.exists():
                data_serie = pd.read_csv(file_path)
                data_serie['date'] = pd.to_datetime(data_serie['date'])
            else:
                data_serie = self.fetch_fred_data(serie)
                data_serie.to_csv(file_path, index=False)

            # Calculamos la última fecha de los datos existentes
            if not data_serie.empty:
                max_date = data_serie['date'].max()
                new_start_date = max_date + pd.DateOffset(days=1)
            else:
                new_start_date = pd.to_datetime('2010-01-01')

            # Solicita solo los datos nuevos si la última fecha es antes de la fecha final requerida
            if new_start_date <= pd.to_datetime(self.end_date_fred_str):
                new_data = self.fetch_fred_data(serie, start_date=new_start_date)
                if not new_data.empty:
                    data_serie = pd.concat([data_serie, new_data])
                    data_serie.to_csv(file_path, index=False)

            if not data_serie.empty:
                self.data = pd.merge_asof(self.data.sort_values('date'), data_serie.sort_values('date'), on='date',
                                     direction="backward")

    def add_mean_reversion_score(self, window=1):
        mean = self.data['close'].rolling(window).mean()
        std = self.data['close'].rolling(window).std()
        self.data[f'mean_reversion_score_{window}m'] = (self.data['close'] - mean) / std

    def add_price_oscillator(self, short_window=1, long_window=2):
        short_ma = self.data['close'].rolling(window=short_window, min_periods=1).mean()
        long_ma = self.data['close'].rolling(window=long_window, min_periods=1).mean()
        self.data['price_oscillator'] = short_ma - long_ma

    def add_on_balance_volume(self):
        self.data['obv'] = (np.sign(self.data['close_or'].diff()) * self.data['volume']).fillna(0).cumsum()

    def add_accumulation_distribution(self):
        clv = ((self.data['close_or'] - self.data['low']) - (self.data['high'] - self.data['close_or'])) / (
                    self.data['high'] - self.data['low'])
        self.data['adl'] = (clv * self.data['volume']).cumsum()

    def add_commodity_channel_index(self, window=1):
        tp = (self.data['high'] + self.data['low'] + self.data['close_or']) / 3
        ma = tp.rolling(window).mean()
        md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        self.data['cci'] = (tp - ma) / (0.015 * md)

    def calculate_all(self):
        """
        Calculamos todos los factores alfa definidos y los añadimos al DataFrame.
        """
        #self.add_historical_volatility()
        self.add_skewness_kurtosis()
        self.add_momentum_indicators()
        self.add_bollinger_bands()
        self.add_fetch_fred()
        self.add_time()
        self.add_mean_reversion_score()
        self.add_price_oscillator()
        self.add_on_balance_volume()
        self.add_accumulation_distribution()
        self.add_commodity_channel_index()
        return self.data

