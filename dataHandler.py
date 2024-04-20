import warnings
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import pandas_ta as ta
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pandas_market_calendars as mcal
warnings.filterwarnings("ignore")


class DataHandler:
    DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]
    DEFAULT_METRICS = ["GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "BOPGSTB", "PPIACO", "UMCSENT", "T10Y2Y", 'TB3MS']
    SAVE_PATH = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/ZenithQuants/daily'

    def __init__(self, etfs=None, metrics=None, start_date='2010-01-01', end_date=None, freq='M', save=False, \
                 start_back='2020-01-02', end_back='2024-04-10', fetch_fred=True, technical=True):
        self.etfs = etfs if etfs else self.DEFAULT_ETFs
        self.metrics = metrics if metrics else self.DEFAULT_METRICS
        self.start_date = start_date
        self.adjusted_start_date = (pd.to_datetime(self.start_date) - DateOffset(months=28)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.freq = freq
        self.scalers = {}
        self.etf_data = {}
        self.save = save
        self.start_back = start_back
        self.end_back = end_back
        start_back_datetime = datetime.strptime(self.start_back, '%Y-%m-%d')
        # Restar un día
        end_date_fred = start_back_datetime - timedelta(days=1)
        # Convertir de nuevo a una cadena en formato 'YYYY-MM-DD'
        self.end_date_fred_str = end_date_fred.strftime('%Y-%m-%d')
        self.API_KEY_FRED = 'd6d360dc757c1441a1a73cf568e60e1c'
        self.fetch_fred = fetch_fred
        self.technical = technical

    def load_data(self):
        lista_df = []
        for etf in self.etfs:
            ticker = yf.Ticker(etf)
            data = ticker.history(start=self.adjusted_start_date, end=self.end_date)
            if isinstance(data.index, pd.DatetimeIndex):
                data.index = data.index.tz_localize(None)
            #data = data[self.start_date:]
            data = data.fillna(method='ffill').fillna(method='bfill')
            # Renombrar las columnas a minúsculas, ya que yfinance usa mayúsculas
            data.columns = data.columns.str.lower()
            data.rename(columns={'stock splits': 'split'}, inplace=True)
            data.rename(columns={'dividends': 'dividend'}, inplace=True)
            data.rename(columns={'capital gains': 'gains'}, inplace=True)
            data['etf'] = etf
            # Asegurarse de que el DataFrame contenga las columnas esperadas por Zipline
            if self.save:
                data_save = data.copy()
                data_save = data_save[self.start_back:self.end_back]
                # Cargar el calendario para NYSE (puedes cambiarlo por el mercado correspondiente a tus ETFs)
                nyse_calendar = mcal.get_calendar('NYSE')
                # Obtener las sesiones de trading en el rango deseado
                trading_days = nyse_calendar.schedule(start_date=self.start_back, end_date=self.end_back)
                trading_days.index = trading_days.index.tz_localize('UTC')
                # Convertir el índice en una columna 'date'
                data_save['date'] = data_save.index
                # Los días faltantes específicos que quieres insertar
                dias_faltantes = [pd.Timestamp('2022-06-20', tz='UTC'), pd.Timestamp('2023-06-19', tz='UTC')]
                for dia in dias_faltantes:
                    # Encuentra el último día de mercado disponible antes del día faltante
                    ultimo_dia_disponible = trading_days[trading_days.index < dia].index[-1].tz_localize(None)
                    # Encuentra los datos para ese último día disponible
                    datos_ultimo_dia = data_save[data_save['date'] == ultimo_dia_disponible]
                    # Copia esos datos para crear una nueva fila para el día faltante
                    nuevos_datos = datos_ultimo_dia.copy()
                    nuevos_datos['date'] = dia.tz_localize(None)  # Actualiza la fecha a la fecha faltante
                    # Añade los nuevos datos al DataFrame
                    data_save = pd.concat([data_save, nuevos_datos])
                # Ordena el DataFrame por fecha después de insertar los nuevos días
                data_save.sort_values(by='date', inplace=True)
                trading_days.index = trading_days.index.tz_localize(None)
                # Rellenar solamente para los días hábiles de trading
                data_save = data_save.reset_index()
                data_save = data_save.loc[:, ['date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split']]
                # Definir el nombre del archivo de salida
                data_save.to_csv(f'{self.SAVE_PATH}/{etf}.csv', index=False)
            data = data[:self.end_date_fred_str]
            data = data.resample(self.freq).last()
            data = data.reset_index()
            data.rename(columns={'Date': 'date'}, inplace=True)
            if self.fetch_fred:
                data = self.fetch_fred_load(data)
            if self.technical:
                data = self.technical_data(data)
            # Filtrar datos para mantener solo aquellos después de self.start_date
            data = data[data['date'] >= pd.to_datetime(self.start_date)]
            #print(data[["close","TB3MS"]].tail())
            #print(data["TB3MS"][-1:])
            lista_df.append(data)
        data_ohlcv = pd.concat(lista_df, ignore_index=True)
        #print(data_ohlcv.columns)
        #print(data_ohlcv.head())
        #print(data_ohlcv.tail())
        return data_ohlcv

    def fetch_fred_data(self, serie):
        url = (
            f"https://api.stlouisfed.org/fred/series/observations?"
            f"series_id={serie}&api_key={self.API_KEY_FRED}"
            f"&file_type=json&observation_start={self.start_date}"
            f"&observation_end={self.end_date_fred_str}"
        )
        #print(url2)
        response = requests.get(url).json()
        #print(response)
        df_fred = pd.DataFrame(response['observations'])
        df_fred['date'] = pd.to_datetime(df_fred['date'])
        df_fred['date'] = df_fred['date'].dt.tz_localize(None)
        df_fred[serie] = pd.to_numeric(df_fred['value'], errors='coerce')
        return df_fred[['date', serie]]

    def fetch_fred_load(self, data):
        data_fred = data.copy()
        for serie in self.DEFAULT_METRICS:
            data_serie = self.fetch_fred_data(serie)
            data_fred = pd.merge_asof(data_fred, data_serie, on='date', direction="backward")
        return data_fred

    def technical_data(self, etf_data):
        etf_data['RSI'] = etf_data.ta.rsi(close='close', length=14)
        etf_data['EMA'] = etf_data.ta.ema(close='close')
        etf_data['SMA'] = etf_data.ta.sma(close='close')
        etf_data['TEMA'] = etf_data.ta.tema(close='close')
        etf_data['CCI'] = etf_data.ta.cci(high='high', low='low', close='close')
        etf_data['CMO'] = etf_data.ta.cmo(close='close')
        etf_data['MACD_signal'] = ta.macd(etf_data['close'])['MACDs_12_26_9']
        etf_data['PPO_signal'] = ta.ppo(etf_data['close'])['PPOs_12_26_9']
        etf_data['ROC'] = etf_data.ta.roc(close='close')
        etf_data['CMF'] = etf_data.ta.cmf(high='high', low='low', close='close', volume='volume')
        etf_data['ADX'] = etf_data.ta.adx(high='high', low='low', close='close')['ADX_14']
        etf_data['HMA'] = etf_data.ta.hma(close='close', length=9)
        # Consider similar refactoring for other indicators...
        etf_data["close"] = etf_data["close"].pct_change(1)
        return etf_data

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
        # Descargar datos
        if etfs is None:
            n_etfs = self.DEFAULT_ETFs
            data = yf.download(self.DEFAULT_ETFs, start=self.start_date, end=self.end_date_fred_str)
        else:
            n_etfs = etfs
            data = yf.download(etfs, start=self.start_date, end=self.end_date_fred_str)

        # Tomar solo el precio ajustado al cierre
        data_portfolio = data["Close"]

        # Remover 'SPY'
        #data_portfolio = data_portfolio.drop(columns=["SPY"])

        # Cambiar el nombre de las columnas a los nombres de los tickers
        # Esto ya debería estar implícito en el DataFrame resultante de yf.download
        # Si necesitas realizar un cambio específico, puedes hacerlo manualmente o de manera programática

        # Si necesitas cambiar el nombre de las columnas de manera programática, aquí te dejo cómo podrías hacerlo:
        # Esto es opcional, dado que `yf.download` ya asigna como nombre de cada columna el ticker correspondiente
        data_portfolio.columns = [col for col in n_etfs if col != "SPY"]
        data_portfolio.tail()

        return data_portfolio