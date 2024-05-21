import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm
import os


class DataAnalyzer:
    def __init__(self, n_features=10, features_file="caracteristicas_seleccionadas.csv",
                 params_file="hiperparametros_seleccionados.csv",
                 evaluations_rmse="evaluations_rmse.csv",
                 evaluations_rmse_m="evaluations_rmse_m.csv",
                 merged_file="merged.csv",
                 portfolio="portfolio_value.csv",
                 benchmark="spy_returns.csv"):
        self.features_file = features_file
        self.params_file = params_file
        self.evaluations_rmse = evaluations_rmse
        self.evaluations_rmse_m = evaluations_rmse_m
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.merged_file = merged_file
        self.n_features = n_features
        self.load_data()

    def load_data(self):
        if os.path.exists(self.features_file):
            self.df_features = pd.read_csv(self.features_file, header=None)
            self.df_features.columns = ['ETF', 'Date', 'Method'] + [f'Feature_{i}' for i in range(self.n_features)]
            self.df_features = self.df_features[self.df_features['Method'] != 'estatico']
            self.df_features['Date'] = pd.to_datetime(self.df_features['Date'], format='%Y-%m-%d', errors='coerce')
        else:
            self.df_features = pd.DataFrame()


        if os.path.exists(self.params_file):
            self.df_params = pd.read_csv(self.params_file)
            self.df_params['date'] = pd.to_datetime(self.df_params['date'], errors='coerce')
        else:
            self.df_params = pd.DataFrame()

        # Load RMSE evaluations
        if os.path.exists(self.evaluations_rmse):
            self.evaluations_rmse = pd.read_csv(self.evaluations_rmse, index_col="ETF")
        else:
            self.evaluations_rmse = pd.DataFrame()

        if os.path.exists(self.evaluations_rmse_m):
            self.evaluations_rmse_m = pd.read_csv(self.evaluations_rmse_m, index_col="Method")
        else:
            self.evaluations_rmse_m = pd.DataFrame()


        if os.path.exists(self.merged_file):
            self.data = pd.read_csv(self.merged_file)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        else:
            self.data = pd.DataFrame()


        if os.path.exists(self.portfolio):
            self.portfolio = pd.read_csv(self.portfolio)
            self.portfolio['date'] = pd.to_datetime(self.portfolio['date'])
            self.portfolio.set_index('date', inplace=True)
        else:
            self.portfolio = pd.DataFrame()


        if os.path.exists(self.benchmark):
            self.benchmark = pd.read_csv(self.benchmark)
            self.benchmark['date'] = pd.to_datetime(self.benchmark['date'])
            self.benchmark.set_index('date', inplace=True)
        else:
            self.benchmark = pd.DataFrame()

    def analyze_portfolio(self):
        # Concatenamos los retornos del portafolio y el benchmark
        returns_data = pd.concat([self.portfolio, self.benchmark], axis=1).dropna()
        returns_data.columns = ['Portfolio', 'Benchmark']

        # Cálculmos la estadísticas financieras
        beta, alpha = self.calculate_beta_alpha(returns_data['Portfolio'], returns_data['Benchmark'])
        sharpe = self.calculate_sharpe(returns_data['Portfolio'])
        sortino = self.calculate_sortino(returns_data['Portfolio'])
        drawdown = self.calculate_drawdown(returns_data['Portfolio'])
        var, cvar = self.calculate_var_cvar(returns_data['Portfolio'])
        print(
            f""" ----------------------------------------------------------- 
            Analyze Portfolio:                                                      
            --------------------------------------------------------------------- 
            Beta : {np.round(beta, 3)} \t Alpha: {np.round(alpha, 3)} \t \ 
            Sharpe: {np.round(sharpe, 3)} \t Sortino: {np.round(sortino, 3)} 
            --------------------------------------------------------------------- 
            VaR : {np.round(var, 3)} \t cVaR: {np.round(cvar, 3)} \t \ 
            VaR/cVaR: {np.round(cvar / var, 3)} \t Drawdown: {np.round(-drawdown.min(), 3)}
            --------------------------------------------------------------------- 
            """)

        self.plot_results(returns_data, drawdown)

    def calculate_beta_alpha(self, portfolio, benchmark):
        covariance = np.cov(portfolio, benchmark)
        beta = covariance[0, 1] / covariance[1, 1]
        alpha = portfolio.mean() - beta * benchmark.mean()
        print(portfolio.mean(), beta , benchmark.mean())
        return beta, alpha

    def calculate_sharpe(self, returns, risk_free_rate=0):
        mean_return = returns.mean() * 12  # Anualización del rendimiento
        std_dev_return = returns.std() * np.sqrt(12)  # Anualización de la desviación estándar
        return (mean_return - risk_free_rate) / std_dev_return

    def calculate_sortino(self, returns, risk_free_rate=0):
        negative_returns = returns[returns < 0]
        negative_std = negative_returns.std() * np.sqrt(12)
        mean_return = returns.mean() * 12
        return (mean_return - risk_free_rate) / negative_std

    def calculate_drawdown(self, returns):
        cumulative_returns = returns.cumsum().apply(np.exp)
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown

    def calculate_var_cvar(self, returns, confidence_level=0.05):
        sorted_returns = sorted(returns)
        index_var = int(confidence_level * len(sorted_returns))
        var = -sorted_returns[index_var]
        cvar = -np.mean(sorted_returns[:index_var])
        return var, cvar

    def plot_results(self, returns_data, drawdown):
        plt.figure(figsize=(15, 7))
        plt.fill_between(drawdown.index, drawdown * 100, 0, color="#9C27B0")
        plt.title('Drawdown', size=15)
        plt.show()


    def plot_feature_frequencies(self):
        colors = sns.color_palette("magma", n_colors=self.n_features)
        feature_counts = \
        self.df_features.melt(id_vars=['ETF', 'Date', 'Method'], value_vars=[f'Feature_{i}' for i in range(self.n_features)],
                              var_name='Feature_Num', value_name='Feature')['Feature'].value_counts().head(self.n_features)
        feature_counts.plot(kind='bar', color=colors)  # Magma Orange
        plt.title('Top 10 Características más Frecuentes')
        plt.ylabel('Frecuencia')
        plt.xlabel('Características')
        plt.xticks(rotation=10)
        plt.savefig('top_features.png')
        plt.show()

    def analyze_parameters(self):
        #descripcion = self.df_params.describe()
        frecuencias = {col: self.df_params[col].value_counts() for col in self.df_params.columns if
                       col not in ['Modelo','Metodo','ETF', 'date']}
        x = len(frecuencias)
        cols = int(np.ceil(np.sqrt(x)))  # Número de columnas (mayor entero que no exceda la raíz cuadrada de x)
        rows = int(np.ceil(x / cols))  # Número de filas

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        axs = axs.flatten()
        parametros = [col for col in self.df_params.columns if col not in ['Modelo','Metodo','ETF', 'date']]
        colors = sns.color_palette("magma", n_colors=len(parametros))
        for i, col in enumerate(parametros):
            self.df_params[col].hist(ax=axs[i], color=colors[i])
            axs[i].set_title(f'Distribución de {col}')
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Frecuencia')
        plt.show()
        return  frecuencias

    def plot_rmse_comparisons(self):
        colors = sns.color_palette("magma", n_colors=4)
        ax = self.evaluations_rmse.plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por ETF y Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('ETF')
        plt.xticks(rotation=45)
        plt.legend(title='Método')
        plt.tight_layout()
        plt.show()

        colors = sns.color_palette("magma", n_colors=len(self.evaluations_rmse_m))
        # Crear el gráfico
        ax = self.evaluations_rmse_m['RMSE'].plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('Métodos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()