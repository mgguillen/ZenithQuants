import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
import matplotlib.dates as mdates
import seaborn as sns
import time
import concurrent.futures
import sys
import pandas_market_calendars as mcal
from dataHandler import DataHandler
from featureSelector import FeatureSelector
from modelBuilder import ModelBuilder
from portfolioOptimizer import PortfolioOptimizer
from backtestVectorizer import BacktestVectorizer
from vectorizedBacktester import VectorizedBacktester
from alphaFactors import AlphaFactors


#Seleccionamos la frecuencia
#freq = "W"
freq = "M"
DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]
inicio = time.time()
bv = VectorizedBacktester( initial_capital=100000, etfs = [etf for etf in DEFAULT_ETFs if etf != 'SPY'])
market_etfs = bv.load_etf_data()
benchmark = bv.load_ben_data()
nyse_calendar = mcal.get_calendar('NYSE')
#'2024-04-10'
# Obtenemos las sesiones de trading en el rango deseado
trading_days = nyse_calendar.schedule(start_date='2020-01-02', end_date='2024-01-04')
trading_days.index = trading_days.index.tz_localize(None)
trading_days_series = trading_days.index.to_series()
if freq == "M":
    first_business_days = trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.month]).first()
elif freq == "W":
    first_business_days = trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.isocalendar().week]).first()
multi_index = pd.MultiIndex.from_product([first_business_days, DEFAULT_ETFs], names=['date', 'ETF'])
# Inicializamos el DataFrame con el MultiIndex
portfolio_value = pd.DataFrame(index=multi_index, columns=['n_acciones', 'Price', 'Balance', 'Capital'])
portfolio_value['n_acciones'] = 0
portfolio_value['Balance'] = 0
portfolio_value['Capital'] = 0
portfolio_value['Price'] = 0
# Inicializamos el Dataframe de inversión
predictions = pd.DataFrame()
current_month = None
# En primer lugar obtenemos el vector de inversion
for today in first_business_days:
    print("today: ", today)
    dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), freq=freq)
    # Cargamos los datos necesarios
    datos = dt.load_data()
    end_date_fred_str = dt.end_date_fred_str
    start_date = dt.start_date
    rend_spy = 0
    #p_TB3MS = datos["TB3MS"].iloc[-1]
    # Una vez que tenemos los datos y como voy a calcular los procesos de seleccion de atributos, seleccion
    # de hiperparametros y prediccion de forma independiente para cada ETF, preparo la función que voy a usar
    # en el cálculo paralelizado
    def procesar_etf(etf):
        global p_TB3MS
        # seleccionamos datos por ETF
        data_etf = datos[datos["etf"] == etf]
        # Añadimos atributos alpha_factors
        alpha_factors = AlphaFactors(data_etf,end_date_fred_str,start_date)
        alpha = alpha_factors.calculate_all()
        # Guardamos el valor mas actual de TB3MS para el cálculo de Sortino en Optimización de la Cartera
        p_TB3MS = alpha["TB3MS"].iloc[-1]
        # Seleccionamos los mejores atributos
        c = FeatureSelector(alpha)
        caracteristicas = c.calculate_feature_importance(method='causal', n_features=10)
        atributos = caracteristicas[['top_features']].iloc[0][0]
        data_model = alpha[["date"] + atributos + ["close"]].reset_index(drop=True)
        # Calculamos los mejores hiperparametros y realizamos la prediccion
        b = ModelBuilder(data_model,model='XGBR')
        rend = b.predict_rend()
        # Devolvemos un Dataframe con los diccionarios de ETF y predicción
        return pd.DataFrame({'date':[today],'ETF': [etf], 'Predict_rend': [rend]})
    # Inicializa una lista para guardar los DataFrames
    predictions_dfs = []
    # Usamos ThreadPoolExecutor para paralelizar el procesamiento
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(procesar_etf, etf) for etf in DEFAULT_ETFs]
        for future in concurrent.futures.as_completed(futures):
            try:
                prediction_etf = future.result()
                # Agregamos el Dataframe a la lista
                predictions_dfs.append(prediction_etf)
            except Exception as exc:
                print(f'Una excepción ocurrió: {exc}')
    # Concatenamos todos los DataFrames en la lista en un solo DataFrame de predictions sin indice
    predictions_ni = pd.concat(predictions_dfs, ignore_index=True)
    # Obtenemos la prediccion de SPY y determinamos nuestra estrategia con respecto a ella
    rend_spy = predictions_ni.loc[predictions_ni['ETF'] == 'SPY', 'Predict_rend'].iloc[0]
    # Calcula Alfa para cada ETF
    predictions_ni['Alfa'] = predictions_ni.apply(lambda row: (row['Predict_rend']) - (rend_spy*1.1) if row['Predict_rend'] > 0 else 0, axis=1)
    # decidimos si comprar o vender de forma que para comprar sera 1 y para vender -1
    predictions_ni['Accion'] = predictions_ni['Alfa'].apply(lambda x: 1 if x > 0 else -1)
    predictions_ni['All'] = 0
    # Filtramos los ETFs cuyo 'Alfa' es mayor que cero y excluye 'SPY'
    etfs_filtered = predictions_ni[(predictions_ni['Alfa'] > 0) & (predictions_ni['ETF'] != 'SPY')]
    # Continuamos si existen predicciones de mejor rendimiento, en caso contrario no hacemos nada
    if len(etfs_filtered) > 0:
        if len(etfs_filtered) > 3:
            etfs_portfolio = etfs_filtered.nlargest(3, 'Alfa')['ETF'].tolist()
        else:
           etfs_portfolio = etfs_filtered['ETF'].tolist()
        # Obtenemos los datos necesarios para la cartera
        datos_portfolio = dt.load_data_portfolio(etfs=etfs_portfolio)
        # Inicializamos la clase para la optimizacion de la cartera y obtenemos los % de inversion
        p = PortfolioOptimizer(data=datos_portfolio, p_etfs=etfs_portfolio, p_beta=False, p_TB3MS=p_TB3MS)
        porcentajes_inversion = p.portfolio_optimize()
        # Añadimos los porcentajes de inversión para cada etf
        predictions_ni['Inversion'] = predictions_ni['ETF'].apply(lambda etf: porcentajes_inversion.get(etf, 0))
        # Guardamos los resultados
        predictions_ni.to_csv('resultados.csv', index=False)
    else:
        predictions_ni['Inversion'] = 0
        predictions_ni['Accion'] = -1
        predictions_ni['All'] = 1
    # Concatenamos las predicciones
    predictions = pd.concat([predictions, predictions_ni], ignore_index=True)
predictions.to_csv('predictions.csv', index=False)
bv.load_price_data(predictions)
bv.backtest_strategy()
# Marca de tiempo al final del proceso
fin = time.time()
duracion = fin - inicio
print(f"El proceso tomó {duracion / 60} minutos, el record esta en 1.08 minutos. Empezamos con 13.66 minutos")
