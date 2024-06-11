import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm
import os


class DataAnalyzer:
    def __init__(self, vfreq = 'M',n_features=10, features_file="caracteristicas_seleccionadas.csv",
                 params_file="hiperparametros_seleccionados.csv",
                 evaluations_rmse="evaluations_rmse.csv",
                 evaluations_rmse_m="evaluations_rmse_m.csv",
                 merged_file="merged.csv",
                 portfolio="portfolio_value.csv",
                 benchmark="spy_returns.csv"):
        self.features_file = features_file
        #self.params_file_template = params_file_template
        self.params_file = "hiperparametros_seleccionados{model}.csv"
        self.models = ['XGBR','LGBMR',  'RFR']  # Nombres de los modelos 'LGBMR',
        self.df_params = {model: pd.read_csv(self.params_file.format(model=model)) for model in self.models}
        #print(self.df_params)
        self.evaluations_rmse = evaluations_rmse
        self.evaluations_rmse_m = evaluations_rmse_m
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.merged_file = merged_file
        self.n_features = int(n_features)
        self.load_data()
        self.vfreq = 12
        if vfreq == 'W':
            self.vfreq = 52
        

    def load_data(self):
        if os.path.exists(self.features_file):
            self.df_features = pd.read_csv(self.features_file, header=None)
            #print(self.n_features, len(self.n_features))
            self.df_features.columns = ['ETF', 'date', 'Method'] + [f'Feature_{i}' for i in range(self.n_features)]
            #self.df_features = self.df_features[self.df_features['Method'] != 'estatico']
            #print(self.df_features[['ETF', 'date', 'Method']].head())
            #self.df_features['date'] = pd.to_datetime(self.df_features['date'], format='%Y-%m-%d', errors='coerce')
            #print(self.df_features[['ETF', 'date', 'Method']].tail())
        else:
            self.df_features = pd.DataFrame()


        #if os.path.exists(self.params_file):
        #    self.df_params = pd.read_csv(self.params_file)
            #self.df_params['date'] = pd.to_datetime(self.df_params['date'], errors='coerce')
        #else:
        #    self.df_params = pd.DataFrame()

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
        mean_return = returns.mean() * self.vfreq  # Anualización del rendimiento
        std_dev_return = returns.std() * np.sqrt(self.vfreq)  # Anualización de la desviación estándar
        return (mean_return - risk_free_rate) / std_dev_return

    def calculate_sortino(self, returns, risk_free_rate=0):
        negative_returns = returns[returns < 0]
        negative_std = negative_returns.std() * np.sqrt(self.vfreq)
        mean_return = returns.mean() * self.vfreq
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
        colors = sns.color_palette("coolwarm", n_colors=self.n_features)
        feature_counts = \
        self.df_features.melt(id_vars=['ETF', 'date', 'Method'], value_vars=[f'Feature_{i}' for i in range(self.n_features)],
                              var_name='Feature_Num', value_name='Feature')['Feature'].value_counts().head(self.n_features)
        #print(feature_counts)
        #feature_counts = feature_counts.groupby(['ETF', 'Feature']).size().unstack(fill_value=0)

        feature_counts.plot(kind='bar', color=colors)  # Magma Orange

        plt.title('Top 10 Características más Frecuentes')
        plt.ylabel('Frecuencia')
        plt.xlabel('Características')
        plt.xticks(rotation=10)
        #plt.savefig('top_features.png')
        plt.show()

    def plot_feature_frequencies_etf(self):
        colors = sns.color_palette("coolwarm", n_colors=self.n_features)
        feature_counts = \
            self.df_features.melt(id_vars=['ETF', 'date', 'Method'],
                                  value_vars=[f'Feature_{i}' for i in range(self.n_features)],
                                  var_name='Feature_Num', value_name='Feature')

        # Group by ETF and Feature to get the counts
        feature_counts = feature_counts.groupby(['ETF', 'Feature']).size().unstack(fill_value=0)

        # Set up the subplots
        n_etfs = feature_counts.shape[0]
        cols = int(np.ceil(np.sqrt(n_etfs)))  # Número de columnas
        rows = int(np.ceil(n_etfs / cols))  # Número de filas

        fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(16, 24), sharex=True)
        fig.subplots_adjust(hspace=0.5)

        # Flatten axs array for easy iteration
        axs = axs.flatten()

        for ax, (etf, counts) in zip(axs, feature_counts.iterrows()):
            counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_title(f'Características más Frecuentes para {etf}', fontsize=8)
            ax.set_ylabel('Frecuencia')
            ax.set_xlabel('Características')
            ax.set_xticklabels(counts.index,  rotation=85, fontsize=8)

        # Hide any unused subplots
        for ax in axs[len(feature_counts):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_feature_rmse(self):
        #caracteristicas_seleccionadas = pd.read_csv('caracteristicas_seleccionadas.csv')
        evaluations_rmse = self.evaluations_rmse # pd.read_csv('evaluations_rmse.csv')
        print(self.df_features)
        feature_counts = \
            self.df_features.melt(id_vars=['ETF', 'date', 'Method'],
                                  value_vars=[f'Feature_{i}' for i in range(self.n_features)],
                                  var_name='Feature_Num', value_name='Feature')
        print(feature_counts)
        # Unir los datos por ETF y fecha
        feature_counts['date'] = pd.to_datetime(feature_counts['date'])
        evaluations_rmse['date'] = pd.to_datetime(evaluations_rmse['date'])
        merged_data = pd.merge(feature_counts, evaluations_rmse, on=['ETF', 'date'], how='inner')
        print(merged_data)

        # Procesar datos para analizar el impacto de las características sobre el RMSE
        df_features_rmse = merged_data[['ETF', 'date', 'RMSE', 'Feature']]
        print(df_features_rmse)
        # Calcular estadísticas descriptivas del RMSE por característica
        feature_rmse_stats = df_features_rmse.groupby('Feature')['RMSE'].agg(['mean', 'std', 'count']).sort_values(
            by='mean')

        # Graficar el impacto de las características sobre el RMSE
        plt.figure(figsize=(14, 7))
        sns.barplot(x=feature_rmse_stats.index, y='mean', data=feature_rmse_stats, palette='coolwarm', ci=None)
        plt.errorbar(x=feature_rmse_stats.index, y=feature_rmse_stats['mean'], yerr=feature_rmse_stats['std'],
                     fmt='none', c='black')
        plt.title('Impacto de las Características sobre el RMSE')
        plt.ylabel('Media del RMSE')
        plt.xlabel('Características')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def analyze_parameters(self):
        fig, axs = plt.subplots(nrows=len(self.models), ncols=1, figsize=(15, 10 * len(self.models)))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for idx, model in enumerate(self.models):
            df = self.df_params[model]
            frecuencias = {col: df[col].value_counts() for col in df.columns if
                           col not in ['Modelo', 'Metodo', 'ETF', 'date']}
            parametros = [col for col in df.columns if col not in ['Modelo', 'Metodo', 'ETF', 'date']]
            x = len(frecuencias)
            cols = int(np.ceil(np.sqrt(x)))  # Número de columnas (mayor entero que no exceda la raíz cuadrada de x)
            rows = int(np.ceil(x / cols))  # Número de filas

            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            axs = axs.flatten()
            colors = sns.color_palette("coolwarm", n_colors=len(parametros))
            for i, col in enumerate(parametros):
                df[col].hist(ax=axs[i], color=colors[i])
                axs[i].set_title(f'Distribución de {col} para {model}')
                axs[i].set_xlabel(col)
                axs[i].set_ylabel('Frecuencia')
            plt.tight_layout()
            plt.show()

    def plot_hyperparam_rmse(self):
        # Unir los datos por ETF y fecha
        #evaluations_rmse = pd.read_csv('evaluations_rmse.csv')
        evaluations_rmse = self.evaluations_rmse.reset_index()
        df_params = self.df_params.reset_index()
        evaluations_rmse['date'] = pd.to_datetime(evaluations_rmse['date'])
        df_params['date'] = pd.to_datetime(df_params['date'])
        print(evaluations_rmse.tail(), df_params.tail())


        merged_data = pd.merge(df_params, evaluations_rmse, on=['ETF', 'date'], how='inner')
        print(merged_data.head())

        # Procesar datos para analizar el impacto de los hiperparámetros sobre el RMSE
        hyperparam_columns = [col for col in df_params.columns if
                              col not in ['ETF', 'date', 'Modelo', 'Metodo']]
        df_hyperparams_rmse = merged_data.melt(id_vars=['ETF', 'date', 'RMSE'], value_vars=hyperparam_columns,
                                               var_name='Hyperparam', value_name='Value')

        # Calcular estadísticas descriptivas del RMSE por hiperparámetro
        hyperparam_rmse_stats = df_hyperparams_rmse.groupby(['ETF','Hyperparam','Value'])['RMSE'].agg(
            ['mean', 'std', 'count']).sort_values(by='mean').reset_index()
        #
        # Configurar los subplots
        etfs = df_hyperparams_rmse['ETF'].unique()
        x = len(etfs)
        #x = len(hyperparam_columns)
        cols = int(np.ceil(np.sqrt(x)))  # Número de columnas (mayor entero que no exceda la raíz cuadrada de x)
        rows = int(np.ceil(x / cols))  # Número de filas

          # Número de columnas en los subplots
        #rows = (n_etfs + cols - 1) // cols  # Número de filas en los subplots

        fig, axs = plt.subplots(cols, rows, figsize=(20, 4 * rows), sharey=True)
        axs = axs.flatten()

        # Graficar el impacto de los hiperparámetros sobre el RMSE para cada ETF
        colors = sns.color_palette("coolwarm", len(hyperparam_columns))
        print('hyper: ',hyperparam_rmse_stats.tail())
        for i, (ax, (etf, etf_data)) in enumerate(zip(axs, hyperparam_rmse_stats.groupby('ETF'))):
            sns.barplot(x='Hyperparam', y='mean', hue='Value', data=etf_data, ax=ax, palette='coolwarm', ci=None)
            #ax.errorbar(x=range(len(etf_data)), y=etf_data['mean'], yerr=etf_data['std'], fmt='none', c='black',
            #            capsize=5)
            ax.set_title(f'Impacto de Hiperparámetros sobre el RMSE para {etf}')
            if i // cols == rows - 1:  # Verificar si está en la última fila
                ax.set_xlabel('Hiperparámetros')
            else:
                ax.set_xlabel('')
            ax.set_ylabel('Media del RMSE')
            ax.tick_params(axis='x', rotation=60)
            ax.legend().set_visible(False)  # Ocultar la leyenda

            # Ocultar subplots vacíos
        for j in range(len(etfs), len(axs)):
            axs[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_model_hyperparam_rmse(self):
        # Unir los datos por ETF y fecha
        merged_data = pd.merge(self.df_hyperparams, self.df_rmse, on=['ETF', 'date'], how='inner')

        # Procesar datos para analizar el impacto de los hiperparámetros sobre el RMSE
        hyperparam_columns = [col for col in self.df_hyperparams.columns if col not in ['ETF', 'date', 'Modelo', 'Metodo']]
        df_hyperparams_rmse = merged_data.melt(id_vars=['ETF', 'date', 'Modelo', 'RMSE'], value_vars=hyperparam_columns,
                                               var_name='Hyperparam', value_name='Value')

        # Calcular estadísticas descriptivas del RMSE por modelo e hiperparámetro
        hyperparam_rmse_stats = df_hyperparams_rmse.groupby(['Modelo', 'Hyperparam', 'Value'])['RMSE'].agg(['mean', 'std', 'count']).reset_index()

        # Graficar el impacto de los hiperparámetros sobre el RMSE por modelo
        g = sns.FacetGrid(hyperparam_rmse_stats, col='Modelo', col_wrap=3, height=4, sharey=False)
        g.map(sns.barplot, 'Hyperparam', 'mean', 'Value', palette='coolwarm', ci=None)
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)
        g.set_titles("{col_name}")
        g.set_axis_labels("Hiperparámetros", "Media del RMSE")
        g.add_legend()
        plt.tight_layout()
        plt.show()



    def plot_rmse_comparisons1(self):
        colors = sns.color_palette("coolwarm", n_colors=4)
        ax = self.evaluations_rmse.plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por ETF y Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('ETF')
        plt.xticks(rotation=45)
        plt.legend(title='Método')
        plt.tight_layout()
        plt.show()

        colors = sns.color_palette("coolwarm", n_colors=len(self.evaluations_rmse_m))
        # Crear el gráfico
        ax = self.evaluations_rmse_m['RMSE'].plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('Métodos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_rmse_comparisons(self):
        # Calcular la media de RMSE por ETF y Método
        rmse_by_etf_method = self.evaluations_rmse.groupby(['ETF', 'Method'])['RMSE'].mean().unstack()
        colors = sns.color_palette("coolwarm", n_colors=rmse_by_etf_method.shape[1])

        # Crear el gráfico de barras por ETF y Método
        ax = rmse_by_etf_method.plot(kind='bar', figsize=(18, 10), width=0.8, color=colors)
        plt.title('Media de RMSE por ETF y Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('ETF')
        plt.xticks(rotation=30)
        plt.legend(title='Método')
        plt.tight_layout()
        plt.show()

        # Calcular la media de RMSE por Método
        rmse_by_method = self.evaluations_rmse.groupby('Method')['RMSE'].mean()
        colors = sns.color_palette("coolwarm", n_colors=rmse_by_method.shape[0])

        # Crear el gráfico de barras por Método
        ax = rmse_by_method.plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('Métodos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()