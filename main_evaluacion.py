
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
from vectorizedBacktester import VectorizedBacktester
from alphaFactors import AlphaFactors
import time
import concurrent.futures
import sys
import pandas_market_calendars as mcal

#freq = "W"
freq = "M"
DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]

inicio = time.time()

#bk = BacktestVectorizer([etf for etf in DEFAULT_ETFs if etf != 'SPY'])
bv = VectorizedBacktester( initial_capital=100000, etfs = [etf for etf in DEFAULT_ETFs if etf != 'SPY'])
market_etfs = bv.load_etf_data()
benchmark = bv.load_ben_data()

nyse_calendar = mcal.get_calendar('NYSE')
#'2024-04-10'
# Obtener las sesiones de trading en el rango deseado
# Lo realizamos con un índice doble, primer dia activo de cada mes, ETF
trading_days = nyse_calendar.schedule(start_date='2020-01-02', end_date='2021-01-04')
trading_days.index = trading_days.index.tz_localize(None)
#active_days = trading_days.index
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
#predictions = pd.DataFrame(columns=[ 'date','ETF','Predict_rend', 'Alfa', 'Accion', 'Inversion'])
evaluations = pd.DataFrame()
current_month = None
# Vamos a obtener predictions el dataframe de inversion

for today in first_business_days:
    print("today: ", today)
    dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), freq=freq)
    datos = dt.load_data()
    end_date_fred_str = dt.end_date_fred_str
    start_date = dt.start_date
    # Cargamos el Dataset con los datos necesarios para nuestra estrategia
    # Indicamos desde cuando tomamos datos y cuando inicia el backtesting, para no usar datos de
    # validación en los entrenamientos, también con el parámetro save, tenemos la posibilidad de
    # guardar los datos en ficheros CSV par su uso en el Backtesting, primero inicializamos la clase
    # y posteriormente aplicamos el metodo load_data()
    rend_spy = 0
    # Guardamos el valor mas actual de TB3MS para el cálculo de Sortino en Optimización de la Cartera
    #p_TB3MS = datos["TB3MS"].iloc[-1]
    # Una vez que tenemos los datos y como voy a calcular los procesos de seleccion de atributos, seleccion
    # de hiperparametros y prediccion de forma independiente para cada ETF, preparo la función que voy a usar
    # en el cálculo paralelizado
    for method in ['shap']:#, 'causal', 'selectkbest']:
    #for method in ['causal']:
        def procesar_etf(etf):
            # seleccionamos datos por ETF
            data_etf = datos[datos["etf"] == etf].copy()
            # print("etf: ######", etf, '\n')
            #print("data_etf",data_etf.tail())

            # Añadimos atributos alpha_factors
            alpha_factors = AlphaFactors(data_etf,end_date_fred_str,start_date)
            #alpha = data_etf
            alpha = alpha_factors.calculate_all().copy()
            #print("alpha",alpha.tail())

            # Seleccionamos los mejores atributos
            c = FeatureSelector(alpha)

            if method == 'shap':
                caracteristicas = c.calculate_feature_importance(method='shap', n_features=10)
                # print("caracteristicas Shap: ", caracteristicas)
            elif method == 'causal':
                caracteristicas = c.calculate_feature_importance(method='causal', n_features=10)
                # print("caracteristicas Causal: ", caracteristicas)
            elif method == 'selectkbest':
                caracteristicas = c.calculate_feature_importance(method='selectkbest', n_features=10)
                #print("caracteristicas Selectkbest: ", caracteristicas)

            # caracteristicas = c.selectfeature()jajaj
            # caracteristicas = c.causal()
            #print("caracteristicas: ",caracteristicas)
            ##caracteristicas = c.select_atributos_shap()
            ##atributos = caracteristicas.iloc[:10][['top_features']].index.tolist()
            atributos = caracteristicas[['top_features']].iloc[0][0]
            #print("atributos",type(atributos),atributos)
            ##atributos = caracteristicas.iloc[:10][['media']].index.tolist()
            data_model = alpha[["date"] + atributos + ["close"]].reset_index(drop=True).copy()
            # print(f"antes modelos {etf} = ", data_model[["date","close"]].tail())
            #print('data_model:',data_model)
            # Calculamos los mejores hiperparametros y realizamos la prediccion
            b = ModelBuilder(data_model, model='XGBR', split=True)
            # rend = b.predict_rend()[0]
            rmse = b.run()
            # print("rend", type(rend), rend)
            # print(f"rend {etf} = ", rend)
            # Devolvemos un Dataframe con los diccionarios de ETF y predicción
            # print(pd.DataFrame({'date':[today],'ETF': [etf], 'Predict_rend': [rend]})[["date", "ETF"]])
            return pd.DataFrame({'date': [today],'Method': [method], 'ETF': [etf], 'RMSE': [rmse]})


        # Inicializa una lista para guardar los DataFrames
        predictions_dfs = []
        # Usamos ThreadPoolExecutor para paralelizar el procesamiento
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(procesar_etf, etf) for etf in DEFAULT_ETFs]
            # print(futures)
            for future in concurrent.futures.as_completed(futures):
                try:
                    prediction_etf = future.result()
                    # print("ETF", prediction_etf.tail())
                    # Agregamos el Dataframe a la lista
                    predictions_dfs.append(prediction_etf)
                except Exception as exc:
                    print(f'Una excepción ocurrió: {exc}')

        # Concatenamos todos los DataFrames en la lista en un solo DataFrame de predictions sin indice
        predictions_ni = pd.concat(predictions_dfs, ignore_index=True)
        # print('predictions_ni', predictions_ni.head(12))
        # Obtenemos la prediccion de SPY y determinamos nuestra estrategia con respecto a ella
        evaluations = pd.concat([evaluations, predictions_ni], ignore_index=True)
    #print("Evaluacion del modelo: ",evaluations.tail())



    #print(predictions_ni.head(15))


    # Marca de tiempo al final del proceso
    fin = time.time()

    duracion = fin - inicio
    print(f"El proceso tomó {duracion / 60} minutos, el record esta en 1.08 minutos. Empezamos con 13.66 minutos")

#print(evaluations.head())
#print(evaluations.tail())



evaluations.to_csv('6-evaluations.csv', index=False)

evaluations_est = evaluations.groupby(['ETF', 'Method']).RMSE.mean().unstack()
evaluations_est2 = evaluations.groupby(['Method']).RMSE.mean()


# Cargar una paleta de seaborn
colors = sns.color_palette("magma", n_colors=3)

# Graficar con esta paleta
ax = evaluations_est.plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
plt.title('Media de RMSE por ETF y Método')
plt.ylabel('Media RMSE')
plt.xlabel('ETF')
plt.xticks(rotation=45)
plt.legend(title='Método')
plt.tight_layout()
plt.show()

ax = evaluations_est2.plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
plt.title('Media de RMSE por Método')
plt.ylabel('Media RMSE')
plt.xlabel('Metodos')
plt.xticks(rotation=45)
plt.legend(title='Método')
plt.tight_layout()
plt.show()
#bv = VectorizedBacktester(predictions,  initial_capital=100000)
#bv.load_price_data(predictions)
#bv.backtest_strategy()

evaluations_est = evaluations_est.reset_index()
evaluations_est.to_csv('6-evaluations_est.csv', index=False)
print(evaluations_est)


evaluations_est2 = evaluations_est2.reset_index()
evaluations_est2.to_csv('6-evaluations_est2.csv', index=False)
print(evaluations_est2)
























