import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_ta as ta
from datetime import datetime
import requests
import nolds
from scipy.stats import skew, kurtosis
import warnings
import warnings
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
import fbm
from hurst import compute_Hc

warnings.filterwarnings("ignore")

class AlphaFactors:
    def __init__(self, data):
        self.data = data
        self.data['close'] = np.log(self.data['close'] / self.data['close'].shift(1))
        #print(self.data.columns)

    def add_historical_volatility(self, window=2):
        """Calculate rolling historical volatility."""
        self.data[f'volatility_{window}'] = self.data['close'].rolling(window).std() * np.sqrt(52)

    def add_skewness(self, window=2):
        """Calculate rolling skewness of returns."""
        self.data[f'skewness_{window}'] = self.data['close'].rolling(window).apply(skew, raw=True)

    def add_kurtosis(self, window=2):
        """Calculate rolling kurtosis of returns."""
        self.data[f'kurtosis_{window}'] = self.data['close'].rolling(window).apply(kurtosis, raw=True)

    def add_ema_diff(self, span_short=12, span_long=26):
        """Calculate the difference between short-term and long-term exponential moving averages."""
        ema_short = self.data['close'].ewm(span=span_short, adjust=False).mean()
        ema_long = self.data['close'].ewm(span=span_long, adjust=False).mean()
        self.data['ema_diff'] = ema_short - ema_long

    def add_momentum(self, windows=[1, 3, 6]):
        """Calculate momentum for multiple windows and the differences between them."""
        #############################################################################################################################
        self.data['SMA'] = self.data.ta.sma(close='close',length = 2)
        self.data["SMA6"] = self.data["close"].rolling(2).mean().shift(1)
        self.data["SMA12"] = self.data["close"].rolling(4).mean().shift(1)
        self.data["SMA24"] = self.data["close"].rolling(6).mean().shift(1)
        self.data["IMP6"] = self.data["SMA12"] - self.data["SMA6"]
        self.data["IMP18"] = self.data["SMA24"] - self.data["SMA6"]
        self.data["IMP12"] = self.data["SMA24"] - self.data["SMA12"]
        mean6 = self.data["IMP6"].mean()
        std6 = self.data["IMP6"].std()
        mean18 = self.data["IMP18"].mean()
        std18 = self.data["IMP18"].std()
        mean12 = self.data["IMP12"].mean()
        std12 = self.data["IMP12"].std()
        self.data["IMP6"] = (self.data["IMP6"] - mean6) / std6
        self.data["IMP18"] = (self.data["IMP18"] - mean18) / std18
        self.data["IMP12"] = (self.data["IMP12"] - mean12) / std12

        #############################################################################################################################
        '''
        for window in windows:
            momentum_key = f'momentum_{window}m'
            self.data[momentum_key] = self.data['close'].pct_change(periods=window) * 100
            self.data[momentum_key].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.data[momentum_key].fillna(0, inplace=True)  # Asumiendo que 0 es un valor neutral adecuado

        for i in range(len(windows) - 1):
            short_window = windows[i]
            long_window = windows[i + 1]
            short_key = f'momentum_{short_window}m'
            long_key = f'momentum_{long_window}m'
            diff_key = f'momentum_diff_{short_window}m_{long_window}m'

            # Rellena NaN justo antes de hacer la resta
            self.data[short_key].fillna(0, inplace=True)
            self.data[long_key].fillna(0, inplace=True)

            # Ahora realiza la resta, asegurándose de que ambos campos están libres de NaN
            self.data[diff_key] = self.data[short_key] - self.data[long_key]
            '''
    def add_rsi(self, window=1):
        """Calculate Relative Strength Index."""
        delta = self.data['close']#.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ma_up = up.rolling(window).mean()
        ma_down = down.rolling(window).mean()
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        self.data[f'rsi_{window}'] = rsi

    def add_macd(self):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        exp1 = self.data['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        self.data['macd'] = macd
        self.data['signal_line'] = macd.ewm(span=9, adjust=False).mean()

    def add_bollinger_bands(self, window=2, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = self.data['close'].rolling(window).mean()
        rolling_std = self.data['close'].rolling(window).std()
        self.data['bollinger_high'] = rolling_mean + (rolling_std * num_std)
        self.data['bollinger_low'] = rolling_mean - (rolling_std * num_std)

    ##################################################################################################################################
        def fetch_fred_data(self, serie, start_date=None):
            # print("self.end_date_fred_str: ",self.start_date, self.end_date_fred_str)
            if start_date is None:
                startdate = self.start_date
            else:
                startdate = start_date.strftime('%Y-%m-%d')
            # fechas#
            # print(startdate, self.end_date_fred_str)
            url = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id={serie}&api_key={self.API_KEY_FRED}"
                f"&file_type=json&observation_start={startdate}"
                f"&observation_end={self.end_date_fred_str}"
            )

            # print("serie: ", serie)
            # print("self.start_date: ", self.start_date)
            # print("self.end_date_fred_str: ", self.end_date_fred_str)
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
                # Opción 1: Devolver un DataFrame vacío
                return pd.DataFrame()
                # Opción 2: Devolver un DataFrame con las fechas pero los valores rellenos con NaN o ceros
                # Puedes generar un rango de fechas basado en self.start_date y self.end_date_fred_str y rellenarlo según necesites.


        def add_fetch_fred_load(self, data):
            for serie in self.DEFAULT_METRICS:
                file_path = self.data_dir / f"{serie}.csv"
                if file_path.exists():
                    data_serie = pd.read_csv(file_path)
                    # Asegúrate de que 'date' es el tipo correcto después de cargar
                    data_serie['date'] = pd.to_datetime(data_serie['date'])
                else:
                    data_serie = self.fetch_fred_data(serie)
                    data_serie.to_csv(file_path, index=False)

                # Determina la última fecha de los datos existentes
                if not data_serie.empty:
                    max_date = data_serie['date'].max()
                    new_start_date = max_date + pd.DateOffset(days=1)
                    # print("calculo new_start_date: ", new_start_date)
                else:
                    new_start_date = pd.to_datetime('2010-01-01')  # Asumir una fecha de inicio si no hay datos

                # Solicita solo los datos nuevos si la última fecha es antes de la fecha final requerida
                if new_start_date <= pd.to_datetime(self.end_date_fred_str):
                    new_data = self.fetch_fred_data(serie, start_date=new_start_date)
                    if not new_data.empty:
                        data_serie = pd.concat([data_serie, new_data])
                        data_serie.to_csv(file_path, index=False)  # Guardar la data actualizada

                if not data_serie.empty:
                    data = pd.merge_asof(data.sort_values('date'), data_serie.sort_values('date'), on='date',
                                         direction="backward")

            return data


        def add_technical_data(self, etf_data):
            etf_data['RSI'] = etf_data.ta.rsi(close='close', length=2)
            etf_data['EMA'] = etf_data.ta.ema(close='close')
            # etf_data['SMA'] = etf_data.ta.sma(close='close')
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
            # etf_data['close'] = np.log(etf_data['close'] / etf_data['close'].shift(1))
            return etf_data

    ##################################################################################################################################

    def calculate_hurstv1(self):
        """
        Calcula el exponente de Hurst usando la biblioteca hurst.

        :param series: pd.Series
            Serie temporal de precios o retornos.
        :return: float
            Exponente de Hurst (H) y el tamaño de la serie (c).
        """
        # La función compute_Hc devuelve el coeficiente de Hurst (H), el valor c de la ecuación de ajuste y los datos del ajuste
        H, c, data = compute_Hc(self.data['open'], kind='price', simplified=True)

        return H
    
    def calculate_hurst(self):
        """
        Calcula el exponente de Hurst usando la biblioteca hurst.

        :param series: pd.Series
            Serie temporal de precios o retornos.
        :return: float
            Exponente de Hurst (H) y el tamaño de la serie (c).
        """
        # La función compute_Hc devuelve el coeficiente de Hurst (H), el valor c de la ecuación de ajuste y los datos del ajuste
        nvals = list(range(10, 20 + 1))
        hurst_coefficient = nolds.hurst_rs(self.data['LogReturns'].values, fit='poly', nvals=nvals)
        return hurst_coefficient


    def predict_next_month_price(self):
        prices = self.data['LogReturns']
        if prices.isnull().any() or (prices <= 0).any():
            raise ValueError("La serie de precios contiene valores no válidos o cero/negativos.")

        H = self.calculate_hurst()
        print("hurst: ", H)
        if np.isnan(H) or H <= 0 or H >= 1:
            raise ValueError("El cálculo de Hurst resultó en un valor no válido.")

        S0 = prices.iloc[-1]
        if S0 <= 0:
            raise ValueError("El último precio conocido es cero o negativo.")

        f = fbm.FBM(n=1, hurst=H, length=1, method='daviesharte')
        fbm_sample = f.fbm()
        print("fbm_sample: ",fbm_sample)
        mu = np.mean(prices)
        sigma = np.std(prices)

        try:
            next_price = S0 * np.exp((mu - 0.5 * sigma ** 2) + sigma * fbm_sample[1])
            self.data['next_LogReturns'] = next_price
        except OverflowError:
            raise OverflowError("Overflow en el cálculo exponencial.")

        print("Final price data:", self.data[['date', 'next_LogReturns', 'LogReturns']].tail())
        #return next_price
    def calculate_all(self):
        """Calculate all defined alpha factors."""
        #self.add_historical_volatility()
        #self.add_skewness()
        #self.add_kurtosis()
        #self.add_ema_diff()
        #self.add_momentum()
        self.add_rsi()
        #self.add_macd()
        #self.add_bollinger_bands()
        #self.predict_next_month_price()
        return self.data