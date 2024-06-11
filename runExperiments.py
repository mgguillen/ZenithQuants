import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
import time
import concurrent.futures
import pandas_market_calendars as mcal
from pathlib import Path
import csv
import itertools

from dataHandler import DataHandler
from alphaFactors import AlphaFactors
from featureSelector import FeatureSelector
from modelBuilder import ModelBuilder
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates
from dataAnalyzer import DataAnalyzer

plt.style.use("seaborn-v0_8")
sns.set_palette("coolwarm")


def main_evaluation():
    DEFAULT_ETFs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX", "SPY"]

    # Define the parameters for the experiments
    rend_options = [1,0]  # , 1,0,
    freq_options = ['M']  # ['M', 'W']
    start_back = '2019-01-03'
    end_back = '2020-01-04'
    train_periods = [3 * 12, 5 * 12]  # ,, 10 * 12, 5 * 12
    val_periods = [1 * 12]  # , 2 * 12
    num_features_options = [10]  # , 20] 10,, 15, 20, 5, 15
    feature_methods = ['selectkbest','shap','causal']#,'causal']  # , 'selectkbest' 'shap',,'selectkbest'
    hyperparam_methods = ['random','optuna']  # ['grid','random','optuna',
    model_names = ['XGBR', 'RFR','LGBMR']  # , 'RFR'


    all_results = []
    for rend, freq, train_period, val_period, num_features, feature_method, hyperparam_method, model_name in itertools.product(
            rend_options, freq_options, train_periods, val_periods, num_features_options, feature_methods,
            hyperparam_methods, model_names):
        results = run_experiment(rend, freq, start_back, end_back, DEFAULT_ETFs, train_period, val_period, num_features,
                                 feature_method, hyperparam_method, model_name)
        all_results.extend(results)

    evaluations_conf = pd.DataFrame(all_results)
    evaluations_conf.to_csv('evaluations_conf.csv', index=False)
    print_evaluations(evaluations_conf)
    plot_analysis(evaluations_conf)


def get_trading_days(calendar_name, start_date, end_date, freq):
    nyse_calendar = mcal.get_calendar(calendar_name)
    trading_days = nyse_calendar.schedule(start_date=start_date, end_date=end_date)
    trading_days.index = trading_days.index.tz_localize(None)
    trading_days_series = trading_days.index.to_series()

    if freq == "M":
        return trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.month]).first()
    else:
        return trading_days_series.groupby([trading_days_series.dt.year, trading_days_series.dt.strftime('%U')]).first()


def run_experiment(rend, freq, start_back, end_back, etfs, train_period, val_period, num_features, feature_method,
                   hyperparam_method, model_name):
    global vrend
    global vfreq

    vrend = rend
    vfreq = freq

    trading_days = get_trading_days('NYSE', start_back, end_back, freq)
    features_file_path = Path('caracteristicas_seleccionadas.csv')
    evaluations = []

    for today in trading_days:
        print("today: ", today)
        evaluate = evaluate_day(today, etfs, features_file_path, train_period, val_period, num_features, feature_method,
                                hyperparam_method, model_name)
        evaluations.extend(evaluate)

    return evaluations


def evaluate_day(today, etfs, features_file_path, train_period, val_period, num_features, feature_method,
                 hyperparam_method, model_name):
    predictions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for etf in etfs:
            dt = DataHandler(start_date='2010-01-01', start_back=today.strftime('%Y-%m-%d'), freq='M')
            datos = dt.load_data()
            future = executor.submit(process_etf, etf, datos, dt, feature_method, today, features_file_path,
                                     train_period, val_period, num_features, hyperparam_method, model_name)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                predictions.extend(result)
            except Exception as exc:
                print(f'Exception in processing: {exc}')
    return predictions


def benchmark(datos, dt, benchmark='SPY'):
    data_etf = datos[datos["etf"] == benchmark].copy()
    alpha_spy = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date, rend=vrend)
    alpha = alpha_spy.calculate_all().copy()
    return alpha['close']


def process_etf(etf, datos, dt, method, today, features_file_path, train_period, val_period, num_features,
                hyperparam_method, model_name):
    data_etf = datos[datos["etf"] == etf].copy()
    datos = dt.load_data()
    spy = benchmark(datos, dt)
    alpha_factors = AlphaFactors(data_etf, dt.end_date_fred_str, dt.start_date, rend=vrend, benchmark=spy)
    alpha = alpha_factors.calculate_all().copy()
    feature_selector = FeatureSelector(alpha)
    attributes = feature_selector.calculate_feature_importance(method=method, n_features=num_features)[['top_features']].iloc[0][0]

    data_model = alpha[["date"] + attributes + ["close"]].reset_index(drop=True).copy()
    model_builder = ModelBuilder(today, data_model, model=model_name, split=True, etf=etf, rand=hyperparam_method)
    rmse, predictions = model_builder.run()

    with features_file_path.open(mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([etf, today, method] + attributes)

    return [{'date': today,'Rendimiento': vrend, 'Features':num_features,'Method': method, 'ETF': etf, 'RMSE': rmse, 'model': model_name,
             'hyperparam_method': hyperparam_method, 'train_period': train_period, 'val_period': val_period}]


def print_evaluations(evaluations):
    print(evaluations.head())
    evaluations_rmse = evaluations.groupby(
        ['date','ETF', 'Method', 'model', 'hyperparam_method', 'train_period', 'val_period']).RMSE.mean()#.unstack()
    evaluations_rmse.to_csv('evaluations_rmse.csv', index=True)

    evaluations_rmse_m = evaluations.groupby(
        ['date','Method', 'model', 'hyperparam_method', 'train_period', 'val_period']).RMSE.mean()
    evaluations_rmse_m.to_csv('evaluations_rmse_m.csv', index=True)


#def plot_analysis(evaluations):
#    plot_rmse_comparisons(evaluations)
#    plot_rmse_distribution(evaluations)


def plot_analysis(evaluations):
    analyzer = DataAnalyzer()
    #analyzer.plot_hyperparam_rmse()
    analyzer.plot_feature_frequencies()
    analyzer.plot_feature_frequencies_etf()
    analyzer.plot_feature_rmse()

    frecuencias = analyzer.analyze_parameters()

    analyzer.plot_rmse_comparisons()

    plot_detailed_heatmap(evaluations)
    #plot_rmse_comparisons(evaluations)
    plot_rmse_comparisons_hyper(evaluations)
    plot_rmse_distribution(evaluations)
    #plot_model_comparison(evaluations)
    #plot_feature_selection_comparison(evaluations)
    #plot_hyperparam_method_comparison(evaluations)
    #plot_train_period_comparison(evaluations)
    #plot_val_period_comparison(evaluations)
    #plot_heatmap_method_model(evaluations)
    #plot_heatmap_train_val_period(evaluations)
    #plot_heatmap_rend_num_features(evaluations)
    #plot_spider_chart(evaluations)
    #plot_andrews_curves(evaluations)
    #plot_parallel_coordinates(evaluations)
    plot_complex_heatmap(evaluations)
    #plot_pairplot(evaluations)
    plot_grouped_barplot(evaluations)



def plot_detailed_heatmap(evaluations):
    pivot_table = evaluations.pivot_table(
        values='RMSE',
        index=['Rendimiento', 'hyperparam_method', 'train_period'],
        columns=['Features', 'model', 'Method'],
        aggfunc='mean'
    )

    pivot_table = pivot_table.round(5)

    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()

    cmap = plt.get_cmap('coolwarm')

    plt.figure(figsize=(18, 12))
    ax = sns.heatmap(pivot_table, annot=True, cmap=cmap, cbar_kws={'label': 'RMSE'}, fmt='.5f',
                     vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray')
    cbar = ax.collections[0].colorbar

    # Establecer los ticks de la barra de color con precisión de 5 decimales
    ticks = np.linspace(vmin, vmax, num=10)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{x:.5f}' for x in ticks])

    # Ajustar el formato de los ticks para evitar la notación científica
    def fmt(x, pos):
        return f'{x:.5f}'

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt))

    plt.title('Heatmap Detallado de RMSE para Multiples Variables', fontsize=12)
    plt.xlabel('Combinación de Método de Selección, Modelo y número de atributos', fontsize=10)
    plt.ylabel('Combinación de Rendimiento, Hiperparámetros y Periodo Entrenamiento', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

def prepare_data_for_boxplot(evaluations):
    # Crear una columna combinada con las diferentes métricas
    evaluations['Combination'] = evaluations.apply(lambda row: f"{row['Rendimiento']}-{row['hyperparam_method']}-{row['train_period']}-{row['Features']}-{row['Method']}-{row['model']}",
                                                   axis=1)
    return evaluations


def plot_rmse_comparisons(evaluations):
    plt.figure(figsize=(20, 12))  # Aumentar el tamaño de la figura
    # Preparar los datos
    evaluations_prepared = prepare_data_for_boxplot(evaluations)

    # Agrupar por Combination y calcular el RMSE promedio
    evaluations_grouped = evaluations_prepared.groupby('Combination').agg({'RMSE': 'mean'}).reset_index()
    #print('plot_rmse_comparisons-evaluations_grouped: ',evaluations_grouped)

    # Crear el boxplot con la nueva columna combinada en el eje X
    sns.boxplot(data=evaluations_grouped,
                x='Combination',
                y='RMSE',
                dodge=True,
                medianprops=dict(color='white'),
                palette='coolwarm')
    plt.title('Comparación de Combinaciones de Métricas')
    plt.xlabel('Combinación de Métricas')
    plt.ylabel('RMSE')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')  # Mover la leyenda fuera del gráfico
    plt.xticks(rotation=60)  # Rotar las etiquetas del eje X para que sean legibles
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()



def plot_rmse_comparisons_hyper(evaluations):
    evaluations_grouped = evaluations.groupby(['hyperparam_method', 'model']).agg({'RMSE': 'mean'}).reset_index()
    #print('plot_rmse_comparisons_hyper-evaluations_grouped: ', evaluations_grouped)
    plt.figure(figsize=(14, 10))  # Aumentar el tamaño de la figura
    sns.boxplot(data=evaluations,
                x='hyperparam_method',
                y='RMSE',
                hue='model',
                dodge=True,
                medianprops=dict(color='white'))
    plt.title('Comparción de Método de Selección de Hiperparámetros y Modelo')
    plt.xlabel('Método Selección Hiperparámetros')
    plt.ylabel('RMSE')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')  # Mover la leyenda fuera del gráfico
    plt.xticks(rotation=45)  # Rotar las etiquetas del eje X
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()

def plot_rmse_distribution(evaluations):
    plt.figure(figsize=(14, 7))
    histplot = sns.histplot(data=evaluations, x='RMSE', hue='model', kde=True, palette='coolwarm')
    plt.title('RMSE Distribution')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')

    # Obtener handles y labels de la gráfica y crear una leyenda manualmente
    histplot.legend_.set_title('Model')
    plt.show()

def plot_model_comparison(evaluations):
    plt.figure(figsize=(14, 7))
    sns.barplot(data=evaluations, x='model', y='RMSE', ci=None, palette='magma')
    plt.title('Average RMSE by Model')
    plt.xlabel('Model')
    plt.ylabel('Average RMSE')
    plt.show()


def plot_feature_selection_comparison(evaluations):
    plt.figure(figsize=(14, 7))
    sns.barplot(data=evaluations, x='Method', y='RMSE', ci=None, palette='magma')
    plt.title('Average RMSE by Feature Selection Method')
    plt.xlabel('Feature Selection Method')
    plt.ylabel('Average RMSE')
    plt.show()


def plot_hyperparam_method_comparison(evaluations):
    plt.figure(figsize=(14, 7))
    sns.barplot(data=evaluations, x='hyperparam_method', y='RMSE', ci=None, palette='magma')
    plt.title('Average RMSE by Hyperparameter Selection Method')
    plt.xlabel('Hyperparameter Selection Method')
    plt.ylabel('Average RMSE')
    plt.show()


def plot_train_period_comparison(evaluations):
    plt.figure(figsize=(14, 7))
    sns.barplot(data=evaluations, x='train_period', y='RMSE', ci=None, palette='magma')
    plt.title('Average RMSE by Training Period')
    plt.xlabel('Training Period')
    plt.ylabel('Average RMSE')
    plt.show()


def plot_val_period_comparison(evaluations):
    plt.figure(figsize=(14, 7))
    sns.barplot(data=evaluations, x='val_period', y='RMSE', ci=None, palette='magma')
    plt.title('Average RMSE by Validation Period')
    plt.xlabel('Validation Period')
    plt.ylabel('Average RMSE')
    plt.show()


def plot_heatmap_method_model(evaluations):
    pivot_table = evaluations.pivot_table(values='RMSE', index='Method', columns='model', aggfunc='mean')
    plt.figure(figsize=(14, 7))
    sns.heatmap(pivot_table, annot=True, cmap="magma", fmt='.4f')
    plt.title('Heatmap of RMSE by Method and Model')
    plt.xlabel('Model')
    plt.ylabel('Feature Selection Method')
    plt.show()


def plot_heatmap_train_val_period(evaluations):
    pivot_table = evaluations.pivot_table(values='RMSE', index='train_period', columns='val_period', aggfunc='mean')
    plt.figure(figsize=(14, 7))
    sns.heatmap(pivot_table, annot=True, cmap="magma", fmt='.4f')
    plt.title('Heatmap of RMSE by Training and Validation Period')
    plt.xlabel('Validation Period')
    plt.ylabel('Training Period')
    plt.show()

def plot_heatmap_rend_num_features(evaluations):
    pivot_table = evaluations.pivot_table(values='RMSE', index='Rendimiento', columns='Features', aggfunc='mean')
    plt.figure(figsize=(14, 7))
    sns.heatmap(pivot_table, annot=True, cmap="magma", fmt='.4f')
    plt.title('Heatmap of RMSE by Rend Option and Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Rend Option')
    plt.show()


def plot_spider_chart(evaluations):
    categories = ['RMSE', 'train_period', 'val_period', 'Features', 'Rendimiento']
    categories = [*categories, categories[0]]

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw=dict(polar=True))

    for model in evaluations['model'].unique():
        values = evaluations[evaluations['model'] == model][categories[:-1]].mean().tolist()
        values += values[:1]
        ax.plot(categories, values, label=model)
        ax.fill(categories, values, alpha=0.1)

    plt.title('Spider Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()



def plot_andrews_curves(evaluations):
    plt.figure(figsize=(14, 7))
    # Seleccionar solo las columnas numéricas
    numeric_cols = evaluations.select_dtypes(include=[np.number])
    # Añadir la columna de categoría (model)
    andrews_curves_data = pd.concat([numeric_cols, evaluations['model']], axis=1)
    andrews_curves(andrews_curves_data, 'model', colormap='magma')
    plt.title('Andrews Curves')
    plt.show()

def plot_parallel_coordinates(evaluations):
    plt.figure(figsize=(14, 7))
    # Seleccionar solo las columnas numéricas y la columna de categoría 'model'
    numeric_cols = evaluations.select_dtypes(include=[np.number])
    # Añadir la columna de categoría (model)
    parallel_coords_data = pd.concat([numeric_cols, evaluations['model']], axis=1)
    parallel_coordinates(parallel_coords_data, class_column='model', colormap='magma')
    plt.title('Parallel Coordinates Plot')
    plt.show()


def plot_pairplot(evaluations):
    # Seleccionar solo las columnas numéricas y la columna de categoría 'model'
    numeric_cols = evaluations.select_dtypes(include=[np.number])
    # Añadir la columna de categoría (model)
    pairplot_data = pd.concat([numeric_cols, evaluations['model']], axis=1)
    sns.pairplot(pairplot_data, hue='model', palette='coolwarm')
    plt.show()

def prepare_data_for_grouped_barplot(evaluations):
    # Crear una columna combinada con las diferentes métricas
    evaluations['Combination'] = evaluations.apply(lambda row: f"{row['Method']}-{row['model']}-{row['hyperparam_method']}-{row['train_period']}-{row['val_period']}", axis=1)
    return evaluations

def plot_grouped_barplot(evaluations):
    plt.figure(figsize=(18, 10))
    # Preparar los datos
    evaluations_prepared = prepare_data_for_grouped_barplot(evaluations)
    # Crear el gráfico de barras agrupadas
    sns.barplot(data=evaluations_prepared, x='Combination', y='RMSE', hue='model', palette='magma', dodge=True)
    plt.title('Bar Plot de RMSE para Varias Métricas')
    plt.xlabel('Combinación de Métricas')
    plt.ylabel('RMSE')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=60)  # Rotar las etiquetas del eje X para que sean legibles
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()


def plot_complex_heatmap4(evaluations):
    # Crear una tabla pivote con múltiples variables en el índice y las columnas
    pivot_table = evaluations.pivot_table(
        values='RMSE',
        index=['Rendimiento', 'hyperparam_method', 'train_period'],  # Añadir más variables en el índice
        columns=['Features', 'model', 'Method'],  # Añadir más variables en las columnas
        aggfunc='mean'
    )

    # Redondear los valores en la tabla pivote a 5 decimales
    pivot_table = pivot_table.round(5)

    # Definir los límites del rango de colores basados en los valores del RMSE
    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()

    # Crear una escala de colores con más detalles
    levels = np.linspace(vmin, vmax, 100)
    norm = BoundaryNorm(levels, ncolors=256)

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(pivot_table, annot=True, cmap="magma", cbar_kws={'label': 'RMSE'}, fmt='.5f', norm=norm)
    cbar = ax.collections[0].colorbar
    # Establecer los ticks de la barra de color con precisión de 5 decimales
    cbar.set_ticks(np.linspace(vmin, vmax, num=10))
    cbar.set_ticklabels([f'{x:.5f}' for x in np.linspace(vmin, vmax, num=10)])

    plt.title('Heatmap de RMSE para Multiples Variables')
    plt.xlabel('Combinación de Método de Selección, Modelo y número de atributos')
    plt.ylabel('Combinación de Rendimiento, Hiperparámetros y Periodo Entrenamiento')
    plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje X para que sean legibles
    plt.yticks(rotation=0)  # Mantener las etiquetas del eje Y horizontales
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()


#from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm


def plot_complex_heatmap1(evaluations):
    # Crear una tabla pivote con múltiples variables en el índice y las columnas
    pivot_table = evaluations.pivot_table(
        values='RMSE',
        index=['Rendimiento', 'hyperparam_method', 'train_period'],  # Añadir más variables en el índice
        columns=['Features', 'model', 'Method'],  # Añadir más variables en las columnas
        aggfunc='mean'
    )

    # Redondear los valores en la tabla pivote a 5 decimales
    pivot_table = pivot_table.round(5)

    # Definir los límites del rango de colores basados en los valores del RMSE
    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()

    # Crear una paleta de colores personalizada con mayor contraste en los valores bajos
    colors = ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]
    n_bins = 100  # número de segmentos
    cmap_name = 'custom_magma'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Crear niveles personalizados para la escala de colores
    levels = np.linspace(vmin, vmax, 100)
    norm = BoundaryNorm(levels, ncolors=256, clip=True)

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(pivot_table, annot=True, cmap=custom_cmap, cbar_kws={'label': 'RMSE'}, fmt='.5f', norm=norm)
    cbar = ax.collections[0].colorbar
    # Establecer los ticks de la barra de color con precisión de 5 decimales
    cbar.set_ticks(np.linspace(vmin, vmax, num=10))
    cbar.set_ticklabels([f'{x:.5f}' for x in np.linspace(vmin, vmax, num=10)])

    plt.title('Heatmap de RMSE para Multiples Variables')
    plt.xlabel('Combinación de Método de Selección, Modelo y número de atributos')
    plt.ylabel('Combinación de Rendimiento, Hiperparámetros y Periodo Entrenamiento')
    plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje X para que sean legibles
    plt.yticks(rotation=0)  # Mantener las etiquetas del eje Y horizontales
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()


def plot_complex_heatmap2(evaluations):
    # Crear una tabla pivote con múltiples variables en el índice y las columnas
    pivot_table = evaluations.pivot_table(
        values='RMSE',
        index=['Rendimiento', 'hyperparam_method', 'train_period'],  # Añadir más variables en el índice
        columns=['Features', 'model', 'Method'],  # Añadir más variables en las columnas
        aggfunc='mean'
    )

    # Redondear los valores en la tabla pivote a 5 decimales
    pivot_table = pivot_table.round(5)

    # Definir los límites del rango de colores basados en los valores del RMSE
    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()

    # Crear una paleta de colores personalizada con mayor contraste en los valores bajos
    colors = ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]
    n_bins = 100  # número de segmentos
    cmap_name = 'custom_magma'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Crear niveles personalizados para la escala de colores
    levels = np.linspace(vmin, vmax, 100)
    norm = BoundaryNorm(levels, ncolors=256, clip=True)

    plt.figure(figsize=(14, 10))  # Aumentar el tamaño de la figura
    ax = sns.heatmap(pivot_table, annot=True, cmap=custom_cmap, cbar_kws={'label': 'RMSE'}, fmt='.5f', norm=norm,
                     linewidths=0.5, linecolor='gray')
    cbar = ax.collections[0].colorbar
    # Establecer los ticks de la barra de color con precisión de 5 decimales
    cbar.set_ticks(np.linspace(vmin, vmax, num=10))
    cbar.set_ticklabels([f'{x:.5f}' for x in np.linspace(vmin, vmax, num=10)])

    plt.title('Heatmap de RMSE para Multiples Variables', fontsize=16)
    plt.xlabel('Combinación de Método de Selección, Modelo y número de atributos', fontsize=14)
    plt.ylabel('Combinación de Rendimiento, Hiperparámetros y Periodo Entrenamiento', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotar las etiquetas del eje X para que sean legibles
    plt.yticks(rotation=0, fontsize=12)  # Mantener las etiquetas del eje Y horizontales
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()


def plot_complex_heatmap(evaluations):
    # Crear una tabla pivote con múltiples variables en el índice y las columnas
    pivot_table = evaluations.pivot_table(
        values='RMSE',
        index=['Rendimiento', 'hyperparam_method', 'train_period'],  # Añadir más variables en el índice
        columns=['Features', 'model', 'Method'],  # Añadir más variables en las columnas
        aggfunc='mean'
    )

    # Redondear los valores en la tabla pivote a 5 decimales
    pivot_table = pivot_table.round(5)

    # Definir los límites del rango de colores basados en los valores del RMSE
    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()

    # Crear una paleta de colores divergente con énfasis en los extremos
    cmap = plt.get_cmap('coolwarm')

    # Crear niveles personalizados para la escala de colores
    levels = np.linspace(vmin, vmax, 100)
    norm = BoundaryNorm(levels, ncolors=256, clip=True)

    plt.figure(figsize=(18, 12))  # Aumentar el tamaño de la figura
    ax = sns.heatmap(pivot_table, annot=True, cmap=cmap, cbar_kws={'label': 'RMSE'}, fmt='.5f', norm=norm,
                     linewidths=0.5, linecolor='gray')
    cbar = ax.collections[0].colorbar
    # Establecer los ticks de la barra de color con precisión de 5 decimales
    cbar.set_ticks(np.linspace(vmin, vmax, num=10))
    cbar.set_ticklabels([f'{x:.5f}' for x in np.linspace(vmin, vmax, num=10)])

    plt.title('Heatmap de RMSE para Multiples Variables', fontsize=12)
    plt.xlabel('Combinación de Método de Selección, Modelo y número de atributos', fontsize=10)
    plt.ylabel('Combinación de Rendimiento, Hiperparámetros y Periodo Entrenamiento', fontsize=10)
    plt.xticks(rotation=30, ha='right', fontsize=10)  # Rotar las etiquetas del eje X para que sean legibles
    plt.yticks(rotation=0, fontsize=10)  # Mantener las etiquetas del eje Y horizontales
    plt.tight_layout()  # Ajustar el diseño para evitar el recorte de etiquetas
    plt.show()

if __name__ == "__main__":
    main_evaluation()
