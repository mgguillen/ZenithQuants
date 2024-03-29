import pandas_datareader.data as web
import pandas_ta as ta
from datetime import datetime
import requests
import nolds
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import pandas_datareader.data as web
import pandas_ta as ta
from datetime import datetime
import requests
import nolds
from scipy.stats import skew, kurtosis


class DataHandler:
    def __init__(self, p_etfs=None, p_metric_econom=None, p_inicio=None, p_fin= None):
        if p_etfs is None:
            self.p_etfs = ['XLE', 'XLB', 'XLI', 'XLK', 'XLF', 'XLP', 'XLY', 'XLV', 'XLU', 'XLRE', 'XLC', 'SPY']
        else:
            self.p_etfs = p_etfs
        if p_metric_econom is None:
            self.p_metric_econom = ["GDP",       # Producto Interno Bruto
                                    "CPIAUCSL",  # Índices de Precios al Consumidor
                                    "FEDFUNDS",  # Tasas de Interés y Política Monetaria de la Reserva Federal
                                    "UNRATE",    # Tasa de Desempleo
                                    "BOPGSTB",   # Balance Comercial y Datos de Comercio Exterior
                                    "PPIACO",    # Índice de Precios al Productor
                                    "UMCSENT",   # Índice de Confianza del Consumidor
                                    "T10Y2Y"     # Yield Curve (Curva de Rendimientos)
                                    ]

        else:
            self.p_metric_econom = p_metric_econom

        self.API_KEY_FRED = 'd6d360dc757c1441a1a73cf568e60e1c'
        if p_inicio is None:
            self.p_inicio = '2010-01-01'
        else:
            self.p_inicio = p_inicio
        if p_fin is None:
            self.p_fin = datetime.now().strftime('%Y-%m-%d')
        else:
            self.p_fin = p_fin

    def load_data(self):
        indice_temporal = pd.date_range(start=self.p_inicio, end=self.p_fin, freq='D', tz='Etc/GMT+4')
        df = pd.DataFrame(index=indice_temporal, data={'Valores': np.random.rand(len(indice_temporal))})
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df['Date'] = df['Date'].dt.tz_localize(None)
        #print(df.dtypes)
        for serie in self.p_metric_econom:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={serie}&api_key={self.API_KEY_FRED}&file_type=json&observation_start={self.p_inicio}"
            response = requests.get(url)
            data_fred = response.json()['observations']
            df_fred = pd.DataFrame(data_fred)[["date", "value"]]
            df_fred.rename(columns={'date': 'Date'}, inplace=True)
            df_fred[serie] = pd.to_numeric(df_fred['value'], errors='coerce')
            df_fred = df_fred.drop('value', axis=1)
            df_fred['Date'] = pd.to_datetime(df_fred['Date'])
            df_fred['Date'] = df_fred['Date'].dt.tz_localize(None)
            #print(df_fred.dtypes)
            df = pd.merge_asof(df, df_fred.loc[:, ["Date", serie]], on='Date',
                                      direction="backward")  # direction='forward')nearest

        df = df.drop('Valores', axis=1)
        #print(df.dtypes)
        #df.set_index('Date', inplace=True)

        #print(df.columns)
        #print(df.head())


        self.etf_data = {}
        lista_df=[]
        for etf in self.p_etfs:
            ticker = yf.Ticker(etf)
            self.etf_data[etf] = ticker.history(start=self.p_inicio, end=self.p_fin)
            self.etf_data[etf].reset_index(inplace=True)
            self.etf_data[etf]['Date'] = self.etf_data[etf]['Date'].dt.tz_localize(None)
            self.etf_data[etf].reset_index(inplace=True)
            self.etf_data[etf] = self.etf_data[etf].drop('index', axis=1)
            #self.etf_data[etf].rename(columns={'index': 'Date'}, inplace=True)
            #self.etf_data[etf]['Date'] = self.etf_data[etf]['Date'].dt.tz_localize(None)

            # Obtener los datos de precios para cada ETF con yfinance
            self.etf_data[etf]["ETF"]=etf
            self.etf_data[etf]['RSI'] = self.etf_data[etf].ta.rsi(close='close', length=14)
            # df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            # EMA (Media Móvil Exponencial)
            self.etf_data[etf]['EMA'] = self.etf_data[etf].ta.ema(close='close')

            # SMA (Media Móvil Simple)
            self.etf_data[etf]['SMA'] = self.etf_data[etf].ta.sma(close='close')

            # TEMA (Triple Media Móvil Exponencial)
            self.etf_data[etf]['TEMA'] = self.etf_data[etf].ta.tema(close='close')

            # CCI (Índice de Canal de Mercancía)
            self.etf_data[etf]['CCI'] = self.etf_data[etf].ta.cci(high='high', low='low', close='close')

            # CMO (Oscilador de Momento Chande)
            self.etf_data[etf]['CMO'] = self.etf_data[etf].ta.cmo(close='close')

            # MACD
            #df['MACD_signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
            self.etf_data[etf]['MACD_signal'] = ta.macd(self.etf_data[etf]['Close'])['MACDs_12_26_9']

            # PPO (Oscilador de Precio Porcentual)
            self.etf_data[etf]['PPO_signal'] = ta.ppo(self.etf_data[etf]['Close'])['PPOs_12_26_9']

            # ROC (Tasa de Cambio)
            self.etf_data[etf]['ROC'] = self.etf_data[etf].ta.roc(close='close')

            # CMF (Flujo Monetario de Chaikin)
            self.etf_data[etf]['CMF'] = self.etf_data[etf].ta.cmf(high='high', low='low', close='close', volume='volume')

            # ADX (Índice Direccional Promedio)
            self.etf_data[etf]['ADX'] = self.etf_data[etf].ta.adx(high='high', low='low', close='close')['ADX_14']

            # HMA (Media Móvil de Hull)
            self.etf_data[etf]['HMA'] = self.etf_data[etf].ta.hma(close='close', length=9)

            # df['HML'] = df.ta.hma(close='close', length=9)

            # SAR Parabolic Sar
            # print(ta.psar(df['High'], df['Low'], df['Close']).loc[:,['PSARl_0.02_0.2','PSARs_0.02_0.2']].head(25))
            psar = ta.psar(self.etf_data[etf]['High'], self.etf_data[etf]['Low'], self.etf_data[etf]['Close'], fillna=0)
            self.etf_data[etf]['SAR'] = psar['PSARl_0.02_0.2'].values + psar['PSARs_0.02_0.2'].values
            self.etf_data[etf]['PSARaf'] = psar['PSARaf_0.02_0.2'].values
            self.etf_data[etf]['PSARr'] = psar['PSARr_0.02_0.2'].values
            #print("antes merge: ",self.etf_data[etf].dtypes)
            #print(type(self.etf_data[etf]))
            #print(self.etf_data[etf].head())
            #self.etf_data[etf].set_index('Date', inplace=True)
            #self.etf_data[etf]=self.etf_data[etf]
            #print(self.etf_data[etf].columns)
            #print(df.columns)

            lista_df.append(pd.merge(self.etf_data[etf], df, on='Date'))

            #print(self.etf_data.head())

        df_metricas = pd.concat(lista_df, ignore_index=True)

        return df_metricas



