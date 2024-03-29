import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectKBest, f_regression


class FeatureSelector:
    def __init__(self, data=None):
        if data is None:
            print("Por favor ingresa un Set de Datos")
        else:
            self.data = data

    def selectfeature(self):
        y = self.data["inc_close"].shift(1)[1:]
        X = self.data.drop(['ETF',"inc_close"], axis=1).iloc[0:-1,:]
        print(self.data.drop(['ETF', "inc_close"], axis=1).iloc[0:-1,:].head())
        print(X.head())
        print(X.tail())
        print(y.head())
        print(y.tail())

        # Seleccionar las 2 mejores características con f_regression
        selector = SelectKBest(score_func=f_regression, k=5)
        X_new = selector.fit_transform(X, y)

        print("Forma original del conjunto de datos:", X.shape)
        print("Forma del conjunto de datos después de la selección de características:", X_new.shape)
        selected_features = self.data.drop(['ETF',"inc_close"], axis=1).columns[selector.get_support()]

        print("Características seleccionadas:", selected_features)
        # Identificar las características seleccionadas
        print("Características seleccionadas:", selector.get_support(indices=True))
        return selected_features
