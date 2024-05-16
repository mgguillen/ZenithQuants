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


def main_evaluation():
    freq = "M"
    DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]
    bv = VectorizedBacktester(initial_capital=100000, etfs=[etf for etf in DEFAULT_ETFs if etf != 'SPY'])
    trading_days = get_trading_days('NYSE', '2021-01-02', '2022-01-04', freq)

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

def evaluate_dayv2(today, etfs, features_file_path):
    dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
    datos = dt.load_data()

    all_predictions = pd.DataFrame()

    #for method in ['shap', 'causal', 'selectkbest', 'estatico']:
    for etf in etfs:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Crear futures para cada ETF
            #futures = {executor.submit(process_etf, etf, datos, dt, method, today, features_file_path): etf for etf in etfs}
            futures = {executor.submit(process_etf, etf, datos, dt, method, today, features_file_path): method for method in
                       ['shap', 'causal', 'selectkbest', 'estatico']}
            # Lista para recoger resultados
            method_predictions = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    method_predictions.append(result)
                except Exception as exc:
                    print(f'{futures[future]} generated an exception: {exc}')

        # Concatenar resultados del método actual
        if method_predictions:
            predictions_df = pd.concat(method_predictions, ignore_index=True)
        all_predictions = pd.concat([all_predictions, predictions_df], ignore_index=True)

    return all_predictions


def evaluate_dayv1(today, etfs, features_file_path):
    dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
    datos = dt.load_data()

    predictions = pd.DataFrame()

    for method in ['shap', 'causal', 'selectkbest', 'estatico']:
        #predictions_dfs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_etf, etf, datos, dt, method, today, features_file_path) for etf in etfs]
            predictions_dfs = [future.result() for future in concurrent.futures.as_completed(futures)]
        #predictions_dfs.extend(process_etf(etf, datos, dt, method, today, features_file_path) for etf in etfs)
        #print(method, predictions_dfs)
        predictions_m = pd.concat(predictions_dfs, ignore_index=True)
        predictions = pd.concat([predictions, predictions_m], ignore_index=True)
    return predictions


def evaluate_day(today, etfs, features_file_path):
    predictions = pd.DataFrame()
    # Mover la creación del pool de hilos al nivel de 'evaluate_day'
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for etf in etfs:
            for method in ['shap', 'causal', 'selectkbest', 'estatico']:
                # Cada tarea recibe su propio conjunto de datos
                dt = DataHandler(start_date='2009-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
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


def process_etf(etf, datos, dt, method, today, features_file_path):
    data_etf = datos[datos["etf"] == etf].copy()
    alpha_factors = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date)
    alpha = alpha_factors.calculate_all().copy()

    feature_selector = FeatureSelector(alpha)
    if method == 'shap':
        caracteristicas = feature_selector.calculate_feature_importance(method='shap', n_features=10)
        attributes = caracteristicas[['top_features']].iloc[0][0]
        # print("caracteristicas Shap: ", caracteristicas)
    elif method == 'causal':
        caracteristicas = feature_selector.calculate_feature_importance(method='causal', n_features=10)
        attributes = caracteristicas[['top_features']].iloc[0][0]
        # print("caracteristicas Causal: ", caracteristicas)
    elif method == 'selectkbest':
        caracteristicas = feature_selector.calculate_feature_importance(method='selectkbest', n_features=10)
        attributes = caracteristicas[['top_features']].iloc[0][0]
        # print("caracteristicas Selectkbest: ", caracteristicas)
    else:
        attributes = ['price_oscillator', 'MACD_3m', 'obv', 'skewness_6m', 'volume', 'skewness_3m'
            , 'bollinger_high_6', 'CPIAUCSL', 'MACD_48m', 'kurtosis_6m']

    data_model = alpha[["date"] + attributes + ["close"]].reset_index(drop=True).copy()
    model_builder = ModelBuilder(data_model, model='XGBR', split=True, etf=etf)
    rmse = model_builder.run()

    # Write selected features to CSV
    with features_file_path.open(mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([etf, today, method] + attributes)
    #print(pd.DataFrame({'date': [today], 'Method': [method], 'ETF': [etf], 'RMSE': [rmse]}))
    return pd.DataFrame({'date': [today], 'Method': [method], 'ETF': [etf], 'RMSE': [rmse]})


def print_evaluations(evaluations):
    evaluations_rmse = evaluations.groupby(['ETF', 'Method']).RMSE.mean().unstack()
    evaluations_rmse.to_csv('evaluations_rmse.csv', index=True)

    evaluations_rmse_m = evaluations.groupby(['Method']).RMSE.mean()
    evaluations_rmse_m.to_csv('evaluations_rmse_m.csv', index=True)

    print(evaluations_rmse)
    print(evaluations_rmse_m)


def plot_analysis():
    analyzer = DataAnalyzer()
    analyzer.plot_feature_frequencies()
    frecuencias = analyzer.analyze_parameters()
    print(frecuencias)
    analyzer.plot_rmse_comparisons()


if __name__ == "__main__":
    main_evaluation()
