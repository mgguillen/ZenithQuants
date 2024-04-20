import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class BacktestVectorizer:
    DATA_DIRECTORY = Path('C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/daily')

    def __init__(self, etfs, benchmark = "SPY"):
        """
        Inicializa la clase con el DataFrame del portafolio.
        :param portfolio_df: DataFrame con las columnas ['ETF', 'Predict_rend', 'Alfa', 'Accion', 'Inversion'].
        """
        # Asumiendo que 'initial_capital' es tu capital inicial para el backtesting
        self.etfs = etfs
        self.initial_capital = 10000
        self.capital = self.initial_capital
        self.benchmark = benchmark

    def load_etf_data(self):
        def load_data(ticker):
            df = pd.read_csv(self.DATA_DIRECTORY / f'{ticker}.csv') #, parse_dates=['date'])
            #df.set_index('date', inplace=True)
            df['ticker'] = ticker
            return df

        self.df = pd.concat([load_data(ticker) for ticker in self.etfs])
        # Simular señales de compra (este paso debería adaptarse para utilizar predicciones reales)
        #self.df['buy_signal'] = np.random.choice([True, False], size=len(self.df))
        # Calcular rendimientos diarios
        #self.df['returns'] = self.df.groupby('ticker')['close'].pct_change()
        # Calcular rendimientos de la estrategia
        #self.df['strategy_returns'] = self.df['returns'] * self.df['buy_signal'].shift(1)

        return self.df

    def load_ben_data(self):
        self.ben_data = pd.read_csv(self.DATA_DIRECTORY / f'{self.benchmark}.csv', parse_dates=['date'])
        #self.ben_data.set_index('date', inplace=True)
        #self.ben_data['returns'] = self.ben_data['close'].pct_change()
        #self.ben_data['cumulative_returns'] = self.ben_data['returns'].cumsum()

        return self.ben_data

    def apply_trading_strategy(self, portfolio_df):
        """
        Aplica la estrategia de trading basada en las acciones y la inversión indicadas en el DataFrame.
        """
        # Crear un DataFrame para registrar el balance de la cartera
        self.portfolio_df = portfolio_df

        portfolio_value = pd.DataFrame(index=self.portfolio_df.index)
        portfolio_value['Balance'] = np.nan

        # Aplicar las acciones de compra o venta
        for date, row in self.portfolio_df.iterrows():
            etf = row['ETF']
            action = row['Accion']
            inversion = row['Inversion']

            # Aquí necesitas un método para obtener el precio de cierre del ETF en 'date'
            # Por simplificación, se omite ese paso. Asumiremos que tienes una función que lo hace.
            # close_price = get_close_price(etf, date)

            # Ejemplo de precio (este valor debería ser obtenido de tus datos)
            close_price = 100

            # Calcular el número de acciones a comprar/vender
            if action == 'comprar':
                num_shares = (self.capital * inversion) // close_price
                self.capital -= num_shares * close_price
            elif action == 'vender':
                # Aquí necesitarías saber cuántas acciones posees para vender
                num_shares = 0  # Esto es solo un placeholder
                self.capital += num_shares * close_price

            portfolio_value.loc[date, 'Balance'] = self.capital

        return portfolio_value

    def plot_portfolio_performance(self, portfolio_value):
        """
        Grafica el rendimiento de la cartera sobre el tiempo.
        :param portfolio_value: DataFrame con el balance de la cartera.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_value.index, portfolio_value['Balance'], marker='o', linestyle='-')
        plt.title('Rendimiento de la Cartera a lo Largo del Tiempo')
        plt.xlabel('Fecha')
        plt.ylabel('Balance de la Cartera')
        plt.grid(True)
        plt.show()