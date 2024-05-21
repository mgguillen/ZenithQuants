import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class VectorizedBacktester:
    DATA_DIRECTORY = 'C:/Users/ManuelGarcia/Ciencia de Datos y AI/TFM/script/Quants/daily'
    def __init__(self, initial_capital=100000, etfs=[], benchmark="SPY"):
        """
        Inicializamos la clase con capital inicial, lista de ETFs y un benchmark para comparar.
        :param initial_capital: Capital inicial para el trading.
        :param etfs: Lista de ETFs a incluir en el backtesting.
        :param benchmark: ETF o índice de referencia para comparación de rendimiento.
        """
        self.price_data_dir = Path(self.DATA_DIRECTORY)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.price_data = pd.DataFrame()
        self.shares_owned = {}
        self.etfs = etfs
        self.benchmark = benchmark

    def load_price_data(self, decisions_df):
        """
        Carga los datos de precios de los ETFs y las decisiones de mercado.
        :param decisions_df: DataFrame con las decisiones de trading que incluye los ETFs necesarios.
        """

        self.decisions_df = decisions_df
        # Cargar datos de precios para cada ETF
        for etf in self.decisions_df['ETF'].unique():
            etf_file = self.price_data_dir / f"{etf}.csv"
            if etf_file.exists():
                df = pd.read_csv(etf_file)
                df['ETF'] = etf
                self.price_data = pd.concat([self.price_data, df])
        self.price_data['date'] = pd.to_datetime(self.price_data['date']).dt.normalize()

    def backtest_strategy(self):
        """
        Realizamos el backtesting de la estrategia basado en el DataFrame de decisiones y los datos de precios cargados.
        Calcula y almacena el rendimiento del portafolio y lo compara con el benchmark.
        """
        # Obtenemos un dataset que une a traves de las fechas y los etfs, las decisiones con los precios
        self.decisions_df['date'] = pd.to_datetime(self.decisions_df['date']).dt.normalize()
        df_merged = pd.merge(self.decisions_df, self.price_data, on=['date', 'ETF'], how='inner')
        df_merged['shares_owned'] = 0
        df_merged['capital'] = 0
        # Ordenar datos por fecha y acción (ventas primero)
        df_merged.sort_values(by=['date', 'Accion'], ascending=[True, True], inplace=True)
        # Procesar cada fecha individualmente
        for date, group in df_merged.groupby('date'):
            print("date vectorized: ", date, self.capital)
            # Procesamos ventas
            for index, row in group[group['Accion'] == -1].iterrows():
                etf = row['ETF']
                shares = self.shares_owned.get(etf, 0)
                if shares > 0:
                    capital = shares * row["close"]
                    self.capital += capital
                    self.shares_owned[etf] = 0  # Todas las acciones de este etf vendidas
                    df_merged.at[index, 'shares_owned'] = 0
                    if df_merged.at[index, 'All'] == 1:
                        df_merged.at[index, 'capital'] = capital
                    else:
                        df_merged.at[index, 'capital'] = 0
            # Procesamos compras
            investment = self.capital
            for index, row in group[group['Accion'] == 1].iterrows():
                etf, investment_ratio = row['ETF'], row['Inversion']
                amount_to_invest = investment * investment_ratio
                self.capital -= amount_to_invest
                shares_to_buy = int(amount_to_invest / row["close"])
                self.shares_owned[etf] = self.shares_owned.get(etf, 0) + shares_to_buy
                df_merged.at[index, 'shares_owned'] = self.shares_owned[etf]
        df_merged.sort_values(by=['date', 'ETF'], ascending=[True, True], inplace=True)
        # Calculamos el valor de la cartera al final de cada periodo
        df_merged['valor_final'] = (df_merged['shares_owned'] * df_merged["close"]) + df_merged['capital']
        # Guardamos el resultado
        df_merged.to_csv('merged.csv', index=False)
        rendimiento_mensual = df_merged[['date', 'valor_final']].groupby('date').sum()
        rendimiento_mensual['valor_final'] = rendimiento_mensual['valor_final'].replace(0, pd.NA)
        rendimiento_mensual = rendimiento_mensual.fillna(method='ffill')
        rendimiento_mensual = rendimiento_mensual.fillna(method='bfill')
        portfolio_log_returns = np.log(rendimiento_mensual / rendimiento_mensual.shift(1))
        portfolio_log_returns.to_csv('portfolio_value.csv', index=True)
        self.portfolio_value = portfolio_log_returns.cumsum().apply(np.exp)

        # Benchmark
        spy_data = df_merged[df_merged['ETF'] == 'SPY'].groupby('date').sum()
        initial_spy_shares = int(self.initial_capital / spy_data["close"].iloc[0])
        spy_value = initial_spy_shares * spy_data["close"]
        spy_log_returns = np.log(spy_value / spy_value.shift(1))
        spy_log_returns.to_csv('spy_returns.csv', index=True)
        self.spy_returns = spy_log_returns.cumsum().apply(np.exp)

        # Graficamos el resultado
        self.plot_results()

    def load_etf_data(self):
        def load_data(etf):
            df = pd.read_csv(self.price_data_dir / f'{etf}.csv')
            df['ETF'] = etf
            return df
        self.df = pd.concat([load_data(etf) for etf in self.etfs])
        return self.df

    def load_ben_data(self):
        self.ben_data = pd.read_csv(self.price_data_dir / f'{self.benchmark}.csv')

    def plot_results(self):
        """
        Grafica los resultados del backtesting en comparación con SPY.
        """
        # Definición de la paleta de colores
        colors = {
            'portfolio': '#FF5733',  # Naranja Magna
            'spy': '#9C27B0'         # Púrpura Magna
        }
        plt.figure(figsize=(14, 7))
        # Rendimiento del portafolio
        plt.plot(self.portfolio_value.index.get_level_values('date'), self.portfolio_value, label='Portafolio', marker='o', linestyle='-', color=colors['portfolio'])
        # Rendimiento de SPY
        plt.plot(self.spy_returns.index.get_level_values('date'), self.spy_returns, label='SPY', marker='x', linestyle='--', color=colors['spy'])
        plt.title('Rendimiento del Portafolio vs. SPY')
        plt.xlabel('Fecha')
        plt.ylabel('Valor del Portafolio')
        plt.legend()
        plt.grid(True)
        plt.show()
