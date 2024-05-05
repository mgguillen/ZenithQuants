
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
#print(plt.style.available)
plt.style.use("seaborn-v0_8")
from dataHandler import DataHandler
from featureSelector import FeatureSelector
from modelBuilder import ModelBuilder
from portfolioOptimizer import PortfolioOptimizer
from backtestVectorizer import BacktestVectorizer
from alphaFactors import AlphaFactors
import time
import concurrent.futures
import sys
import pandas_market_calendars as mcal

DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]

inicio = time.time()

bk = BacktestVectorizer([etf for etf in DEFAULT_ETFs if etf != 'SPY'])
market_etfs = bk.load_etf_data()
benchmark = bk.load_ben_data()

nyse_calendar = mcal.get_calendar('NYSE')
#'2024-04-10'
# Obtener las sesiones de trading en el rango deseado
trading_days = nyse_calendar.schedule(start_date='2020-01-02', end_date='2020-04-10')
trading_days.index = trading_days.index.tz_localize(None)
active_days = trading_days.index
trading_days_series = trading_days.index.to_series()

first_business_days = trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.month]).first()

#print(first_business_days)

################################################################################################
#multi_index = pd.MultiIndex.from_product([active_days, [etf for etf in DEFAULT_ETFs]], names=['date', 'ETF'])

# Crear DataFrame con el multi-índice
#portfolio_value = pd.DataFrame(index=multi_index, columns=['n_acciones','Price','Balance','Capital'])
#print(portfolio_value.head(20))

multi_index = pd.MultiIndex.from_product([first_business_days, DEFAULT_ETFs], names=['date', 'ETF'])

# Inicializar el DataFrame con el MultiIndex
portfolio_value = pd.DataFrame(index=multi_index, columns=['n_acciones', 'Price', 'Balance', 'Capital'])
#print(portfolio_value2.head(20))


# Inicializar las columnas
portfolio_value['n_acciones'] = 0
portfolio_value['Balance'] = 0
portfolio_value['Capital'] = 0
portfolio_value['Price'] = 0
################################################################################################
#portfolio_value = pd.DataFrame(columns=["ETF","n_acciones","Balance"] ,index=trading_days.index)
#portfolio_value["ETF"] = np.nan
#portfolio_value["n_acciones"] = 0
#portfolio_value["Balance"] = 0
#portfolio_value = portfolio_value.rename_axis('date')
#print(portfolio_value.dtypes)
current_month = None
for today in active_days:

    #print("today month", today.month)
    if current_month is None or current_month != today.month:
        print("today: ", today)
        predictions = pd.DataFrame(columns=['ETF', 'Predict_rend', 'Alfa', 'Accion', 'Inversion'])
        current_month = today.month
        dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), save=False)
        #etfs = dt.DEFAULT_ETFs
        #print(market_etfs.shape)
        #print(market_etfs.shape)
        #print(market_etfs.head())
        #print(market_etfs.tail())
        #print(benchmark.head())

        # Simular la lógica de cálculo del DataFrame para el día actual
        # Este sería un buen lugar para llamar a self.load_data() o cualquier otro
        # método que prepare tus datos de predicción y decisiones de trading para el día actual
        datos = dt.load_data()  # ['2010-01-01':]
        #print(datos.head())
        #resultados = pd.DataFrame(columns=['ETF', 'Predict_rend'])
        rend_spy = 0
        p_TB3MS = datos["TB3MS"].iloc[-1]


        def procesar_etf(etf):
            data_etf = datos[datos["etf"] == etf]
            alpha_factors = AlphaFactors(data_etf)
            alpha = alpha_factors.calculate_all()

            c = FeatureSelector(alpha)
            caracteristicas = c.select_atributos_shap()
            atributos = caracteristicas.iloc[:10][['media']].index.tolist()
            data_model = data_etf[["date"] + atributos + ["close"]].reset_index(drop=True)
            b = ModelBuilder(data_model)
            rend = b.predict_rend()[0]
            # print(f"rend {etf} = ", rend)
            return pd.DataFrame({'ETF': [etf], 'Predict_rend': [rend]})


        # Inicializa una lista para guardar los DataFrames
        predictions_dfs = []

        # Usamos ThreadPoolExecutor para paralelizar el procesamiento
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(procesar_etf, etf) for etf in DEFAULT_ETFs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    prediction_etf = future.result()
                    # En lugar de usar append, agregamos el DataFrame a la lista
                    predictions_dfs.append(prediction_etf)
                except Exception as exc:
                    print(f'Una excepción ocurrió: {exc}')

        # Concatenamos todos los DataFrames en la lista en un solo DataFrame
        predictions = pd.concat(predictions_dfs, ignore_index=True)

        #print("datos:", predictions.tail())
        #print(predictions.columns)

        rend_spy = predictions.loc[predictions['ETF'] == 'SPY', 'Predict_rend'].iloc[0]

        # Calcula Alfa para cada ETF
        predictions['Alfa'] = predictions.apply(
            lambda row: (row['Predict_rend'] ) - rend_spy if row['Predict_rend'] > 0 else 0, axis=1)
        #+ (rend_spy / 100)

        etfs_portfolio = predictions[predictions["Alfa"] > 0].sort_values('Alfa', ascending=False)
        etfs_portfolio = etfs_portfolio["ETF"].index.tolist()

        # decidimos si comprar o vender un etf si es mayor que cero
        predictions['Accion'] = predictions['Alfa'].apply(lambda x: 'comprar' if x > 0 else 'vender')

        # Filtramos los ETFs cuyo 'Alfa' es mayor que cero y excluye 'SPY'
        etfs_portfolio = predictions[(predictions['Alfa'] > 0) & (predictions['ETF'] != 'SPY')]['ETF'].tolist()
        print("##### lista de etfs en la cartera ####")
        print(etfs_portfolio)
        print(len(etfs_portfolio))

        if len(etfs_portfolio) > 0:

            datos_portfolio = dt.load_data_portfolio(etfs=etfs_portfolio)

            p = PortfolioOptimizer(data=datos_portfolio, p_etfs=etfs_portfolio, p_beta=False, p_TB3MS=p_TB3MS)

            porcentajes_inversion = p.portfolio_optimize()

            # añadimos los porcentajes de inversión para cada etf
            predictions['Inversion'] = predictions['ETF'].apply(lambda etf: porcentajes_inversion.get(etf, 0))

            # print(resultados)
            predictions.to_csv('resultados.csv', index=False)
            print("predictions: ", predictions)

            portfolio_value = bk.apply_trading_strategy(predictions, market_etfs, benchmark, today, portfolio_value)
        else:
            portfolio_value.ffill(inplace=True)
        #print(type(trading_days.index))
        #print(type(today))
        #print(type(capital))

        #portfolio_value.at[today, 'Balance'] = capital

        #portfolio_value.iloc[]

        print("results: ", portfolio_value.loc[(today, slice(None))])



                # Marca de tiempo al final del proceso
        fin = time.time()

        # Cálculo y impresión de la duración del proceso
        duracion = fin - inicio
        print(f"El proceso tomó {duracion / 60} minutos, el record esta en 1.08 minutos. Empezamos con 13.66 minutos")

#portfolio_grafica = portfolio_value.reset_index()
#portfolio_grafica = portfolio_grafica[["date", "Capital"]].unique()
portfolio_valueni = portfolio_value.copy()
portfolio_valueni = portfolio_valueni.reset_index()
portfolio_valueni.to_csv('portfolio_value.csv', index=False)
bk.plot_portfolio_performance(portfolio_value)

