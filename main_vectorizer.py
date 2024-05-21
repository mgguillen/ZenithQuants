import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import concurrent.futures
import pandas_market_calendars as mcal
from dataHandler import DataHandler
from alphaFactors import AlphaFactors
from featureSelector import FeatureSelector
from modelBuilder import ModelBuilder
from portfolioOptimizer import PortfolioOptimizer
from vectorizedBacktester import VectorizedBacktester
from dataAnalyzer import DataAnalyzer
plt.style.use("seaborn-v0_8")


def main_vectorizer(rend=1, freq='M', start_back='2023-01-02', end_back='2024-01-04'):
    global vrend
    global vfreq
    vrend = rend
    vfreq = freq
    DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]
    bv = VectorizedBacktester(initial_capital=100000, etfs=[etf for etf in DEFAULT_ETFs if etf != 'SPY'])
    trading_days = get_trading_days('NYSE',  start_back, end_back, vfreq)
    predictions = pd.DataFrame()
    start_time = time.time()
    for today in trading_days:
        print('Today: ', today)
        predictions_ni = process_day(today, DEFAULT_ETFs)
        dt = DataHandler(start_date='2010-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
        if vrend == 0:
            rend_spy = predictions_ni.loc[predictions_ni['ETF'] == 'SPY', 'Predict_rend'].iloc[0]
            # Calcula Alfa para cada ETF
            predictions_ni['Alfa'] = predictions_ni.apply(
                lambda row: (row['Predict_rend']) - (rend_spy * 1.1) if row['Predict_rend'] > 0 else 0, axis=1)
            # decidimos si comprar o vender de forma que para comprar sera 1 y para vender -1
            predictions_ni['Accion'] = predictions_ni['Alfa'].apply(lambda x: 1 if x > 0 else -1)
            predictions_ni['All'] = 0
            # Filtramos los ETFs cuyo 'Alfa' es mayor que cero y excluye 'SPY'
            etfs_filtered = predictions_ni[(predictions_ni['Alfa'] > 0) & (predictions_ni['ETF'] != 'SPY')]
        else:
            rend_spy = 0
            predictions_ni['Alfa'] = predictions_ni.apply(
                lambda row: (row['Predict_rend']) if row['Predict_rend'] > 0 else 0, axis=1)
            predictions_ni['Accion'] = predictions_ni['Alfa'].apply(lambda x: 1 if x > 0 else -1)
            predictions_ni['All'] = 0
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
            p_TB3MS = predictions_ni["TB3MS"].iloc[-1]
            optimizer = PortfolioOptimizer(data=datos_portfolio, p_etfs=etfs_portfolio, p_beta=False, p_TB3MS=p_TB3MS)
            porcentajes_inversion = optimizer.portfolio_optimize()
            # Añadimos los porcentajes de inversión para cada etf
            predictions_ni['Inversion'] = predictions_ni['ETF'].apply(lambda etf: porcentajes_inversion.get(etf, 0))
        else:
            predictions_ni['Inversion'] = 0
            predictions_ni['Accion'] = -1
            predictions_ni['All'] = 1
        # Concatenamos las predicciones
        predictions = pd.concat([predictions, predictions_ni], ignore_index=True)
    predictions.to_csv('predictions.csv', index=False)
    bv.load_price_data(predictions)
    bv.backtest_strategy()
    analyzer = DataAnalyzer()
    analyzer.analyze_portfolio()
    print_process_duration(start_time)


def get_trading_days(calendar_name, start_date, end_date, freq):
    nyse_calendar = mcal.get_calendar(calendar_name)
    trading_days = nyse_calendar.schedule(start_date=start_date, end_date=end_date)
    trading_days.index = trading_days.index.tz_localize(None)
    trading_days_series = trading_days.index.to_series()
    return trading_days_series.groupby([trading_days_series.dt.year,
                                        trading_days_series.dt.month]).first() if freq == "M" else trading_days_series.groupby(
        [trading_days_series.dt.year, trading_days_series.dt.isocalendar().week]).first()


def process_day(today, etfs):
    dt = DataHandler(start_date='2010-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
    datos = dt.load_data()
    spy = benchmark(datos, dt)
    # Creación de hilos para cálculo en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_etf, etf, datos, dt, spy) for etf in etfs]
        predictions_dfs = [future.result() for future in concurrent.futures.as_completed(futures)]
    return pd.concat(predictions_dfs, ignore_index=True)


def benchmark(datos, dt, benchmark = 'SPY'):
    data_etf = datos[datos["etf"] == benchmark].copy()
    alpha_spy = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date,rend=0)
    alpha = alpha_spy.calculate_all().copy()
    return alpha['close']


def process_etf(etf, datos, dt, spy):
    data_etf = datos[datos["etf"] == etf].copy()
    alpha_factors = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date, rend=vrend, benchmark=spy)
    alpha = alpha_factors.calculate_all().copy()
    feature_selector = FeatureSelector(alpha)
    caracteristicas = feature_selector.calculate_feature_importance(method='causal', n_features=10)
    attributes = caracteristicas[['top_features']].iloc[0][0]
    data_model = alpha[["date"] + attributes + ["close"]].reset_index(drop=True).copy()
    p_TB3MS = alpha["TB3MS"].iloc[-1]
    model_builder = ModelBuilder(data_model, model='XGBR', etf=etf)
    rend = model_builder.predict_rend()
    return pd.DataFrame({'date': [dt.start_back], 'ETF': [etf], 'Predict_rend': [rend], 'TB3MS': [p_TB3MS]})


def print_process_duration(start_time):
    duration = (time.time() - start_time) / 60
    print(f"El proceso tomó {duration} minutos.")


if __name__ == "__main__":
    main_vectorizer(rend=1, freq='W')
