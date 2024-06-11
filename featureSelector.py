import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
import causalml
from causalml.inference.meta import LRSRegressor, BaseSRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class FeatureSelector:
    def __init__(self, data):
        """
        Inicializamos la clase y prepara los datos para análisis, vamos a implementar
        3 métodos de selección de variables SHAP, Causal-ml con SHAP y Selectkbest.
        :param data: DataFrame con los datos de entrada.
        """
        self.data = data.fillna(method='ffill').fillna(method='bfill')
        #print(type(self.data))
        self.data = self.data.fillna(0)
        self.X = self.data.drop(['date', 'etf', 'close','close_or'], axis=1).shift(1)
        #print(self.X.shape)
        self.X = self.X.iloc[1:]
        self.y = self.data['close']
        self.y = self.y.iloc[1:]
        self.etf = self.data['etf'].unique()

    def calculate_feature_importance(self, method='causal', n_features=10):
        """
        Calculamos la importancia de las características utilizando alguno de los métodos.
        :param method: Método para calcular la importancia ('shap', 'causal', 'selectkbest').
        :param n_features: Número de características más importantes a seleccionar.
        :return: DataFrame con las características más importantes que vienen de AlphaFactor.
        """
        if method == 'shap':
            return self.select_features_shap(n_features)
        elif method == 'causal':
            return self.causal_method(n_features)
        elif method == 'selectkbest':
            return self.select_k_best(n_features)
        else:
            raise ValueError("Unknown method")

    def select_features_shap(self, n_features):
        """
        Seleccionamos las características más importantes utilizando SHAP.
        :param n_features: Número de características más importantes a seleccionar.
        :return: DataFrame con las características más importantes según SHAP.
        """
        model = XGBRegressor(random_state=42)
        model.fit(self.X, self.y)
        explainer = shap.Explainer(model)
        shap_values = explainer(self.X)
        shap_sum = np.abs(shap_values.values).mean(axis=0)
        features_shap = pd.DataFrame(list(zip(self.X.columns, shap_sum)),
                                     columns=['Feature', 'SHAP Importance'])
        top_features = features_shap.nlargest(n_features, 'SHAP Importance')
        
        top_features_list = top_features['Feature'].tolist()

        #print("shap: ", top_features)
        return pd.DataFrame({'method': 'shap', 'ETF': self.etf, 'top_features': [list(top_features['Feature'])]})

    def causal_method(self, n_features):
        """
        Seleccionamos las características más importantes utilizando Causal-ml y SHAP.
        :param n_features: Número de características más importantes a seleccionar.
        :return: DataFrame con las características más importantes según el método causal.
        """
        tratamiento = np.random.choice([0, 1], size=self.X.shape[0], p=[0.5, 0.5])
        self.X['treatment'] = tratamiento
        model = LGBMRegressor(random_state=42, force_col_wise=True, verbose=-1)
        model.fit(self.X, self.y)
        # Usamos BaseSRegressor con LGBMRegressor como modelo base
        xgb = BaseSRegressor(learner=model)
        xgb.fit(self.X, tratamiento, self.y)
        # Usamos SHAP para explicar las predicciones del modelo
        explainer = shap.Explainer(xgb.model)
        shap_values = explainer.shap_values(self.X)
        shap_sum = np.abs(shap_values).mean(axis=0)
        features_shap = pd.DataFrame(list(zip(self.X.columns, shap_sum)),
                                     columns=['Feature', 'SHAP Importance'])
        features_shap = features_shap[features_shap['Feature'] != 'treatment']
        top_features = features_shap.nlargest(n_features, 'SHAP Importance')
        # Filtrar los valores SHAP para mostrar solo las características más importantes
        # Filtrar los valores SHAP para mostrar solo las características más importantes


        return pd.DataFrame({'method': 'causal', 'ETF': self.etf, 'top_features': [list(top_features['Feature'])]})

    def select_k_best(self, n_features):
        """
        Seleccionamos las características más importantes utilizando el método SelectKBest.
        :param n_features: Número de características más importantes a seleccionar.
        :return: DataFrame con las características más importantes según SelectKBest.
        """
        #print(self.X.head())
        #print(self.X.tail())
        column_names = self.X.columns  # Guarda los nombres de las columnas antes de la imputación
        imputer = IterativeImputer()
        self.X = imputer.fit_transform(self.X)
        self.X = pd.DataFrame(self.X, columns=column_names)
        selector = SelectKBest(score_func=f_regression, k=n_features)
        selected_indices = selector.fit(self.X, self.y).get_support(indices=True)
        selected_features = self.X.columns[selected_indices]
        return pd.DataFrame({'method': 'selectkbest', 'ETF': self.etf, 'top_features': [list(selected_features)]})

