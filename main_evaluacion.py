import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import concurrent.futures
import pandas_market_calendars as mcal
from pathlib import Path
import csv

from dataHandler import DataHandler
from alphaFactors import AlphaFactors
from featureSelector import FeatureSelector
from modelBuilder import ModelBuilder
from vectorizedBacktester import VectorizedBacktester
from dataAnalyzer import DataAnalyzer

plt.style.use("seaborn-v0_8")


def main_evaluation(rend=1, freq='M', start_back='2020-01-02', end_back='2021-01-04' ):
    global vrend
    global vfreq

    vrend = rend
    vfreq = freq

    DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]
    bv = VectorizedBacktester(initial_capital=100000, etfs=[etf for etf in DEFAULT_ETFs if etf != 'SPY'])
    trading_days = get_trading_days('NYSE', start_back, end_back, freq)

    features_file_path = Path('caracteristicas_seleccionadas.csv')
    hyperparameters_file_path = Path('hiperparametros_seleccionados.csv')

    evaluations = pd.DataFrame()

    for today in trading_days:
        print("today: ", today)
        evaluate = evaluate_day(today, DEFAULT_ETFs, features_file_path)
        evaluations = pd.concat([evaluations, evaluate], ignore_index=True)

    evaluations.to_csv('evaluations.csv', index=False)
    print_evaluations(evaluations)
    plot_analysis()


def get_trading_days(calendar_name, start_date, end_date, freq):
    nyse_calendar = mcal.get_calendar(calendar_name)
    trading_days = nyse_calendar.schedule(start_date=start_date, end_date=end_date)
    trading_days.index = trading_days.index.tz_localize(None)
    trading_days_series = trading_days.index.to_series()
    if freq == "M":
        return trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.month]).first()
    else:
        return trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.isocalendar().week]).first()


def evaluate_day(today, etfs, features_file_path):
    predictions = pd.DataFrame()
    # Creamos los hilos para cálculo en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for etf in etfs:
            for method in ['shap', 'causal', 'selectkbest']:
                # Cada tarea recibe su propio conjunto de datos
                dt = DataHandler(start_date='2010-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
                datos = dt.load_data()
                future = executor.submit(process_etf, etf, datos, dt, method, today, features_file_path)
                futures.append(future)

        # Recolectar resultados
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                predictions = pd.concat([predictions, result], ignore_index=True)
            except Exception as exc:
                print(f'Exception in processing: {exc}')
    return predictions


def benchmark(datos, dt, benchmark='SPY'):
    data_etf = datos[datos["etf"] == benchmark].copy()
    alpha_spy = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date,rend=0)
    alpha = alpha_spy.calculate_all().copy()
    return alpha['close']


def process_etf(etf, datos, dt, method, today, features_file_path):
    data_etf = datos[datos["etf"] == etf].copy()
    datos = dt.load_data()
    spy = benchmark(datos, dt)
    alpha_factors = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date, rend=vrend, benchmark=spy)
    alpha = alpha_factors.calculate_all().copy()
    feature_selector = FeatureSelector(alpha)
    if method == 'shap':
        caracteristicas = feature_selector.calculate_feature_importance(method='shap', n_features=10)
        attributes = caracteristicas[['top_features']].iloc[0][0]
    elif method == 'causal':
        caracteristicas = feature_selector.calculate_feature_importance(method='causal', n_features=10)
        attributes = caracteristicas[['top_features']].iloc[0][0]
    elif method == 'selectkbest':
        caracteristicas = feature_selector.calculate_feature_importance(method='selectkbest', n_features=10)
        attributes = caracteristicas[['top_features']].iloc[0][0]
    else:
        attributes = ['price_oscillator', 'MACD_3m', 'obv', 'skewness_6m', 'volume', 'skewness_3m'
            , 'bollinger_high_6', 'CPIAUCSL', 'MACD_48m', 'kurtosis_6m']
    data_model = alpha[["date"] + attributes + ["close"]].reset_index(drop=True).copy()
    model_builder = ModelBuilder(data_model, model='LGBMR', split=True, etf=etf)
    rmse = model_builder.run()
    # Guardamos las mejores caracteristicas
    with features_file_path.open(mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([etf, today, method] + attributes)
    return pd.DataFrame({'date': [today], 'Method': [method], 'ETF': [etf], 'RMSE': [rmse]})


def print_evaluations(evaluations):
    # Guardamos los rmse de los métodos de seleccion
    evaluations_rmse = evaluations.groupby(['ETF', 'Method']).RMSE.mean().unstack()
    evaluations_rmse.to_csv('evaluations_rmse.csv', index=True)

    evaluations_rmse_m = evaluations.groupby(['Method']).RMSE.mean()
    evaluations_rmse_m.to_csv('evaluations_rmse_m.csv', index=True)

def plot_analysis():
    analyzer = DataAnalyzer()
    analyzer.plot_feature_frequencies()
    frecuencias = analyzer.analyze_parameters()
    analyzer.plot_rmse_comparisons()


if __name__ == "__main__":
    main_evaluation(rend=0, freq='M', start_back='2019-01-02', end_back='2020-01-04')
