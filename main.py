
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
import time
import concurrent.futures
import sys
import pandas_market_calendars as mcal

inicio = time.time()

nyse_calendar = mcal.get_calendar('NYSE')

# Obtener las sesiones de trading en el rango deseado
trading_days = nyse_calendar.schedule(start_date='2020-01-02', end_date='2024-04-10')
trading_days.index = trading_days.index.tz_localize(None)
active_days = trading_days.index
for today in active_days:
    print("today: ",today)
    dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), save=False)
    bk = BacktestVectorizer([etf for etf in dt.DEFAULT_ETFs if etf != 'SPY'])
    market_etfs = bk.load_etf_data()
    benchmark = bk.load_ben_data()
    print(market_etfs.head())
    print(market_etfs.tail())
    print(benchmark.head())

    # Simular la lógica de cálculo del DataFrame para el día actual
    # Este sería un buen lugar para llamar a self.load_data() o cualquier otro
    # método que prepare tus datos de predicción y decisiones de trading para el día actual
    datos = dt.load_data()#['2010-01-01':]
    #etfs = ['SPY'] + [etf for etf in dt.DEFAULT_ETFs if etf != 'SPY']
    etfs = dt.DEFAULT_ETFs # ['SPY'] + [etf for etf in dt.DEFAULT_ETFs if etf != 'SPY']
    #datos = dt.fetch_fred_load(datos)
    #print(datos.columns)
    resultados = pd.DataFrame(columns=['ETF', 'Predict_rend'])
    rend_spy = 0
    p_TB3MS = datos["TB3MS"].iloc[-1]

    def procesar_etf(etf):
        data_etf = datos[datos["etf"] == etf]
        c = FeatureSelector(data_etf)
        caracteristicas = c.select_atributos_shap()
        atributos = caracteristicas.iloc[:10][['media']].index.tolist()
        data_model = data_etf[["date"] + atributos + ["close"]].reset_index(drop=True)
        b = ModelBuilder(data_model)
        rend = b.predict_rend()[0]
        #print(f"rend {etf} = ", rend)
        return pd.DataFrame({'ETF': [etf], 'Predict_rend': [rend]})

    # Inicializa una lista para guardar los DataFrames
    resultados_dfs = []

    # Usamos ThreadPoolExecutor para paralelizar el procesamiento
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(procesar_etf, etf) for etf in etfs]
        for future in concurrent.futures.as_completed(futures):
            try:
                resultado_etf = future.result()
                # En lugar de usar append, agregamos el DataFrame a la lista
                resultados_dfs.append(resultado_etf)
            except Exception as exc:
                print(f'Una excepción ocurrió: {exc}')

    # Concatenamos todos los DataFrames en la lista en un solo DataFrame
    resultados = pd.concat(resultados_dfs, ignore_index=True)

    rend_spy = resultados.loc[resultados['ETF'] == 'SPY', 'Predict_rend'].iloc[0]

    # Calcula Alfa para cada ETF
    resultados['Alfa'] = resultados.apply(
        lambda row: (row['Predict_rend'] + (rend_spy / 10)) - rend_spy if row['Predict_rend'] > 0 else 0, axis=1)

    etfs_portfolio = resultados[resultados["Alfa"] > 0].sort_values('Alfa', ascending=False)
    etfs_portfolio = etfs_portfolio["ETF"].index.tolist()

    # Añade una nueva columna basada en la condición de 'Alfa'
    resultados['Accion'] = resultados['Alfa'].apply(lambda x: 'comprar' if x > 0 else 'vender')

    # Filtra los ETFs cuyo 'Alfa' es mayor que cero y excluye 'SPY'
    etfs_portfolio = resultados[(resultados['Alfa'] > 0) & (resultados['ETF'] != 'SPY')]['ETF'].tolist()

    datos_portfolio = dt.load_data_portfolio(etfs = etfs_portfolio)

    #print(f"Valor de p_TB3MS antes de pasar al constructor: {p_TB3MS}")

    p = PortfolioOptimizer(data = datos_portfolio, p_etfs= etfs_portfolio,p_beta = False,p_TB3MS = p_TB3MS)

    porcentajes_inversion = p.portfolio_optimize()

    # Añadir la columna 'Porcentaje Inversión' al DataFrame 'resultados'
    resultados['Inversion'] = resultados['ETF'].apply(lambda etf: porcentajes_inversion.get(etf, 0))

    #print(resultados)
    resultados.to_csv('resultados.csv', index=False)


    # Marca de tiempo al final del proceso
    fin = time.time()

    # Cálculo y impresión de la duración del proceso
    duracion = fin - inicio
    print(f"El proceso tomó {duracion/60} minutos, el record esta en 1.49 minutos. Empezamos con 13.66 minutos")