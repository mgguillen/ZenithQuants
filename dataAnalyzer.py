import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class DataAnalyzer:
    def __init__(self,n_features = 10, features_file = "caracteristicas_seleccionadas.csv",
                 params_file = "hiperparametros_seleccionados.csv",
                 evaluations_rmse = "evaluations_rmse.csv",
                 evaluations_rmse_m = "evaluations_rmse_m.csv"):
        self.features_file = features_file
        self.params_file = params_file
        self.evaluations_rmse = evaluations_rmse
        self.evaluations_rmse_m = evaluations_rmse_m
        self.n_features = n_features
        self.load_data()


    def load_data(self):
        self.df_features = pd.read_csv(self.features_file, header=None)
        self.df_features.columns = ['ETF', 'Date', 'Method'] + [f'Feature_{i}' for i in range(self.n_features)]
        self.df_features = self.df_features[self.df_features['Method'] != 'estatico']
        self.df_features['Date'] = pd.to_datetime(self.df_features['Date'], format='%Y-%m-%d', errors='coerce')
        #self.df_features['Date'] = pd.to_datetime(self.df_features['Date'])

        self.df_params = pd.read_csv(self.params_file)
        self.df_params['date'] = pd.to_datetime(self.df_params['date'], errors='coerce')

        self.evaluations_rmse = pd.read_csv(self.evaluations_rmse, index_col="ETF")
        self.evaluations_rmse_m = pd.read_csv(self.evaluations_rmse_m, index_col="Method")

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
        rows = int(np.ceil(x / cols))  # Número de filas (total de plots dividido por el número de columnas)

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        axs = axs.flatten()
        parametros = [col for col in self.df_params.columns if col not in ['Modelo','Metodo','ETF', 'date']]
        #parametros = ['subsample', 'reg_lambda', 'reg_alpha', 'n_estimators', 'max_depth', 'learning_rate', 'gamma',
        #              'colsample_bytree']
        colors = sns.color_palette("magma", n_colors=len(parametros))
        for i, col in enumerate(parametros):
            #print(i,col)
            self.df_params[col].hist(ax=axs[i], color=colors[i])  # Magma Purple
            axs[i].set_title(f'Distribución de {col}')
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Frecuencia')


        plt.show()
        return  frecuencias #descripcion,

    def plot_rmse_comparisons(self):
        colors = sns.color_palette("magma", n_colors=4)
        #print("evaluations_rmse: ", self.evaluations_rmse)
        ax = self.evaluations_rmse.plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por ETF y Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('ETF')
        plt.xticks(rotation=45)
        plt.legend(title='Método')
        plt.tight_layout()
        plt.show()

        #print("evaluations_rmse_m: ",self.evaluations_rmse_m)
        colors = sns.color_palette("magma", n_colors=len(self.evaluations_rmse_m))

        # Crear el gráfico
        ax = self.evaluations_rmse_m['RMSE'].plot(kind='bar', figsize=(14, 7), width=0.8, color=colors)
        plt.title('Media de RMSE por Método')
        plt.ylabel('Media RMSE')
        plt.xlabel('Métodos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()