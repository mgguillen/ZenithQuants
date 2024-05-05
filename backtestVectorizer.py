import pandas as pd
from pandas.tseries.offsets import BDay
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

        self.initial_capital = 100000
        self.capital = self.initial_capital
        self.capital_spy = self.initial_capital
        self.etfs = etfs
        self.benchmark = benchmark
        self.portfolio_value = pd.DataFrame(columns=['date', 'Balance'])
        self.has_ordered = False
        self.amonth = ""

    def load_etf_data(self):
        def load_data(etf):
            df = pd.read_csv(self.DATA_DIRECTORY / f'{etf}.csv') #, parse_dates=['date'])
            #df.set_index('date', inplace=True)
            df['ETF'] = etf
            return df

        self.df = pd.concat([load_data(etf) for etf in self.etfs])
        # Simular señales de compra (este paso debería adaptarse para utilizar predicciones reales)
        #self.df['buy_signal'] = np.random.choice([True, False], size=len(self.df))
        # Calcular rendimientos diarios
        #self.df['returns'] = self.df.groupby('ticker')['close'].pct_change()
        # Calcular rendimientos de la estrategia
        #self.df['strategy_returns'] = self.df['returns'] * self.df['buy_signal'].shift(1)

        return self.df

    def load_ben_data(self):
        self.ben_data = pd.read_csv(self.DATA_DIRECTORY / f'{self.benchmark}.csv')#, parse_dates=['date'])
        #self.ben_data.set_index('date', inplace=True)
        #self.ben_data['returns'] = self.ben_data['close'].pct_change()
        #self.ben_data['cumulative_returns'] = self.ben_data['returns'].cumsum()

        return self.ben_data

    def benchmark_performance(self, benchmark, today, portfolio_value):
        """Actualiza el rendimiento de SPY en el portfolio_value DataFrame."""
        close_price_spy = benchmark[(benchmark["date"] == today.strftime('%Y-%m-%d'))]["close"].iloc[0]
        # Asumiendo que tienes una columna 'n_acciones' para SPY similar a otros ETFs
        if 1==0:#self.has_ordered:
            num_shares_spy = portfolio_value.loc[(self.amonth, "SPY"), 'n_acciones']
            self.capital_spy += num_shares_spy * close_price_spy
            portfolio_value.at[(today,"SPY"), 'n_acciones'] = 0
            portfolio_value.at[(today,"SPY"), 'Balance'] = num_shares_spy * close_price_spy  # Asumiendo que 'Balance' acumula las ganancias/pérdidas

        #capital_compra = self.capital_spy

        if not self.has_ordered:
            num_shares_spy = int((self.capital_spy * 1) // close_price_spy)
            self.capital_spy -= num_shares_spy * close_price_spy
            portfolio_value.at[(today, "SPY"), 'n_acciones'] = num_shares_spy  # Asumiendo que se acumulan las compras
            portfolio_value.at[(today, "SPY"), 'Balance'] = -1 * num_shares_spy * close_price_spy
            portfolio_value.at[(today, "SPY"), 'Price'] = close_price_spy
            #portfolio_value.at[(today, "SPY"), 'Capital'] =  self.capital_spy #num_shares_spy * close_price_spy
        else:
            Balance_spy = portfolio_value.loc[(self.amonth, "SPY"), 'Balance']
            Capital_spy = portfolio_value.loc[(self.amonth, "SPY"), 'Capital'] + Balance_spy
            Price_spy = portfolio_value.at[(self.amonth, "SPY"), 'Price']
            shares_spy = portfolio_value.loc[(self.amonth, "SPY"), 'n_acciones']
            if self.capital_spy >= close_price_spy:
                num_shares = int((self.capital_spy * 1) // close_price_spy)
                num_shares_spy = shares_spy + num_shares
                Capital_inv = num_shares * close_price_spy
                incremento = (num_shares_spy * close_price_spy) - (shares_spy * Price_spy)
                self.capital_spy -= Capital_inv
                portfolio_value.at[(today, "SPY"), 'n_acciones'] = num_shares_spy
                portfolio_value.at[(today, "SPY"), 'Balance'] = -1 * num_shares_spy * close_price_spy
                portfolio_value.at[(today, "SPY"), 'Price'] = close_price_spy
                #portfolio_value.at[(today, "SPY"), 'Capital'] = self.capital_spy  # num_shares_spy * close_price_spy
            else:
                num_shares_spy = shares_spy
                incremento = (num_shares_spy * close_price_spy) - (shares_spy * Price_spy)
                #self.capital_spy += incremento + Capital_spy
                portfolio_value.at[(today, "SPY"), 'n_acciones'] = num_shares_spy
                portfolio_value.at[(today, "SPY"), 'Balance'] = -1 * num_shares_spy * close_price_spy
                portfolio_value.at[(today, "SPY"), 'Price'] = close_price_spy
                #portfolio_value.at[(today, "SPY"), 'Capital'] = self.capital_spy  # num_shares_spy * close_price_spy



        #self.capital_spy -= num_shares_spy * close_price_spy
        #portfolio_value.at[(today, "SPY"), 'n_acciones'] = num_shares_spy  # Asumiendo que se acumulan las compras
        #portfolio_value.at[(today, "SPY"), 'Balance'] = -1 * num_shares_spy * close_price_spy

        #num_shares_spy = portfolio_value.loc[(self.amonth, "SPY"), 'n_acciones']
        #portfolio_value.at[(today, "SPY"), 'Balance'] = num_shares_spy * close_price_spy
        #return self.capital_spy


    def apply_trading_strategy(self, portfolio_df, market_etfs, benchmark, today, portfolio_value):
        """
        Aplica la estrategia de trading basada en las acciones y la inversión indicadas en el DataFrame.
        """
        # Crear un DataFrame para registrar el balance de la cartera
        self.portfolio_df = portfolio_df
        #self.portfolio_value.loc["date"] = today
        #print("back: ",market_etfs.head())
        #self.amonth = ""


        #portfolio_value = pd.DataFrame(index=self.portfolio_df.index)
        #self.portfolio_value['Balance'] = np.nan

        ventas = []
        compras = []

        # Separar las acciones de venta y compra
        for _, row in portfolio_df.iterrows():
            etf = row['ETF']
            if etf != "SPY":
                action = row['Accion']
                inversion = row['Inversion']
                close_price = \
                market_etfs[(market_etfs["ETF"] == etf) & (market_etfs["date"] == today.strftime('%Y-%m-%d'))]["close"].iloc[0]

                if self.has_ordered and action == 'vender' and portfolio_value.loc[(self.amonth, etf),'n_acciones']>0:
                    # Obtenemos el número de acciones a vender
                    num_shares = portfolio_value.loc[(self.amonth, etf), 'n_acciones']
                    # Si se tienen acciones para vender, se agregan a la lista de ventas
                    #if num_shares > 0:
                    ventas.append((etf, num_shares, close_price))

                if action == 'comprar':
                    # Las compras se agregan a una lista para procesarlas después
                    compras.append((etf, inversion, close_price))
        #print("ventas: ", ventas)
        #######################################################################################################################
        if len(ventas) > 1:
            for etf, num_shares, close_price in ventas:
                Balance = portfolio_value.loc[(self.amonth, etf), 'Balance']
                Capital = portfolio_value.loc[(self.amonth, etf), 'Capital'] + Balance
                Price = portfolio_value.loc[(self.amonth, etf), 'Price']
                shares = portfolio_value.loc[(self.amonth, etf), 'n_acciones']

                self.capital += (num_shares * close_price) #- (shares * Price)
                print("capital Venta: ",self.capital)
                portfolio_value.at[(today, etf), 'n_acciones'] = num_shares
                portfolio_value.at[(today,etf), 'Balance'] =  num_shares * close_price
                portfolio_value.at[(today, etf), 'Price'] = close_price
                # Asumiendo que 'Balance' acumula las ganancias/pérdidas
                #portfolio_value.at[(today, etf), 'Capital'] = self.capital

        #capital_compra = self.capital
        print("capital_compra", self.capital)
        print("compras: ", compras)
        capital_compra = self.capital
        for etf, inversion, close_price in compras:
            if not self.has_ordered:
                num_shares = int((capital_compra * inversion) // close_price)
                self.capital -= num_shares * close_price
                portfolio_value.at[(today, etf), 'n_acciones'] = num_shares
                portfolio_value.at[(today, etf), 'Balance'] =  -1 *num_shares * close_price
                portfolio_value.at[(today, etf), 'Price'] = close_price

            else:
                #Balance = portfolio_value.loc[(self.amonth, etf), 'Balance']
                Capital = portfolio_value.loc[(self.amonth, etf), 'Capital'] #+ Balance
                Price = portfolio_value.loc[(self.amonth, etf), 'Price']
                shares = portfolio_value.loc[(self.amonth, etf), 'n_acciones']
                print("shares: ",shares)

                if self.capital >= close_price and shares != 0:
                    num_shares = int((capital_compra * inversion) // close_price)
                    num_shares_etf = shares + num_shares
                    Capital_inv = num_shares * close_price
                    incremento = (num_shares_etf * close_price) - (shares * Price)
                    self.capital -=  Capital_inv
                    portfolio_value.at[(today, etf), 'n_acciones'] = num_shares_etf
                    portfolio_value.at[(today, etf), 'Balance'] = -1 * num_shares_etf * close_price
                    portfolio_value.at[(today, etf), 'Price'] = close_price
                    # portfolio_value.at[(today, "SPY"), 'Capital'] = self.capital_spy  # num_shares_spy * close_price_spy
                elif self.capital < close_price and shares != 0:
                    num_shares_etf = shares
                    incremento = (num_shares_etf * close_price) - (shares * Price)
                    #self.capital += incremento + Capital
                    portfolio_value.at[(today, etf), 'n_acciones'] = num_shares_etf
                    portfolio_value.at[(today, etf), 'Balance'] = -1 * num_shares_etf * close_price
                    portfolio_value.at[(today, etf), 'Price'] = close_price
                elif shares == 0:
                    num_shares = int((capital_compra * inversion) // close_price)
                    self.capital -= num_shares * close_price
                    portfolio_value.at[(today, etf), 'n_acciones'] = num_shares
                    portfolio_value.at[(today, etf), 'Balance'] = -1 * num_shares * close_price
                    portfolio_value.at[(today, etf), 'Price'] = close_price
            #num_shares = (capital_compra * inversion) // close_price
            #self.capital -= num_shares * close_price
            #portfolio_value.at[(today, etf), 'n_acciones'] = num_shares  # Asumiendo que se acumulan las compras
            #portfolio_value.at[(today, etf), 'Balance'] = -1 * num_shares * close_price
        #self.portfolio_value.loc['Balance'] = self.capital
        #######################################################################################################################
        '''
        if len(ventas) > 1:
            for etf, num_shares, close_price in ventas:
                self.capital += num_shares * close_price
                portfolio_value.at[(today, etf), 'n_acciones'] = 0
                portfolio_value.at[(today,etf), 'Balance'] = num_shares * close_price  # Asumiendo que 'Balance' acumula las ganancias/pérdidas

        capital_compra = self.capital
        print("capital_compra", capital_compra)
        print("compras: ", compras)
        for etf, inversion, close_price in compras:
            num_shares = (capital_compra * inversion) // close_price
            self.capital -= num_shares * close_price
            portfolio_value.at[(today, etf), 'n_acciones'] = num_shares  # Asumiendo que se acumulan las compras
            portfolio_value.at[(today, etf), 'Balance'] = -1 * num_shares * close_price
        #self.portfolio_value.loc['Balance'] = self.capital
        '''
        self.benchmark_performance(benchmark, today, portfolio_value)

        self.has_ordered = True
        self.amonth = today.strftime('%Y-%m-%d')
        portfolio_value.at[(today), 'Capital'] = self.capital
        portfolio_value.at[(today,"SPY"), 'Capital'] = self.capital_spy
        return portfolio_value
    def plot_portfolio_performance(self, portfolio_value):
        """
        Grafica el rendimiento de la cartera y el benchmark SPY sobre el tiempo.
        :param portfolio_value: DataFrame con el balance de la cartera y SPY.
        """
        plt.figure(figsize=(14, 7))

        # Rendimiento de la cartera
        filtered_portfolio_value = portfolio_value[~portfolio_value.index.get_level_values('ETF').isin(['SPY'])]
        # Una vez filtrado ya podemos calcular el Balance
        portfolio_balance = filtered_portfolio_value.groupby(level=0)['Balance'].sum() * -1
        plt.plot(portfolio_balance.index, portfolio_balance, marker='o', linestyle='-', label='Cartera')

        # Rendimiento de SPY como benchmark
        if "SPY" in portfolio_value.index.get_level_values(1):
            spy_balance = portfolio_value.xs('SPY', level=1)['Balance']*-1
            plt.plot(spy_balance.index, spy_balance, marker='x', linestyle='--', label='SPY Benchmark')

        plt.title('Rendimiento de la Cartera vs. SPY Benchmark')
        plt.xlabel('Fecha')
        plt.ylabel('Balance de la Cartera / Benchmark')
        plt.legend()
        plt.grid(True)
        plt.show()




    def plot_portfolio_performancev1(self, portfolio_value):
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