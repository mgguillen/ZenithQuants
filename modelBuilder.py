import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
import numpy as np


class ModelBuilder:
    def __init__(self, data, split = False):
        self.data = data
        self.split = split
        self.row_month = 0

    def prepare_data(self):
        self.y = self.data["close"]#[1:]
        self.X = self.data.drop(["close"], axis=1)

        self.X['date'] = pd.to_datetime(self.X['date'])

        # Encuentra la última fecha en tu DataFrame
        fecha_fin = self.X['date'].max()

        # Calcula la fecha de inicio para el conjunto de prueba (un año antes de la fecha_fin)
        fecha_inicio_prueba = fecha_fin - pd.DateOffset(years=1)

        # Encuentra el índice donde comienza el conjunto de prueba
        indice_split = self.X.index[self.X['date'] >= fecha_inicio_prueba].min()
        #print("self.y antes : ", self.y.head())
        #print("self.y antes : ", self.y.tail())
        #print("self.X antes : ", self.X.tail())
        self.row_month = self.X.iloc[[-1]].drop(['date'], axis=1)
        self.X = self.X.drop(['date'], axis=1).shift(1)#.iloc[1:, :]
        #print("self.X despues : ", self.X.tail())
        if self.split:
            # División de los datos en conjuntos de entrenamiento y prueba
            #print(self.X.tail())
            #print(self.y.tail())
            self.X_train = self.X.iloc[:indice_split-1]
            self.y_train = self.y.iloc[:indice_split-1]
            self.X_test = self.X.iloc[indice_split:]
            self.y_test = self.y.iloc[indice_split:]
        else:
            self.X_train = self.X
            self.y_train = self.y
        #print(self.y_test.tail())

        # X_train, X_test, y_train, y_test

    def build_and_tune_modelv1(self):
        modelo = XGBRegressor()
        param_grid = {
            'n_estimators': [25,50],#,100, 200],
            'learning_rate': [0.01],#, 0.01, 0.1],
            'max_depth': [3, 7],
            'subsample': [0.6, 1],
            'colsample_bytree': [0.6, 0.8],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error',
                                   verbose=1, n_jobs=1)
        grid_search.fit(self.X_train, self.y_train)

        #print("Mejores parámetros:", grid_search.best_params_)
        self.best_model = grid_search.best_estimator_

    def build_and_tune_model(self):
        modelo = LGBMRegressor(verbose=-1)
        param_grid = {
            'n_estimators': [50,100,150],  # Número de árboles de boosting
            'learning_rate': [0.01, 0.05],  # Tasa de aprendizaje
            'num_leaves': [31],  # Número de hojas en un árbol
            'feature_fraction': [0.6, 0.8, 1.0],  # Submuestreo de las características
            'bagging_fraction': [0.8],  # Submuestreo de los datos
            'max_depth': [3],  # Profundidad máxima de los árboles
        }
        # Usamos TimeSeriesSplit para la validación cruzada en series temporales
        tscv = TimeSeriesSplit(n_splits=3)

        # Configuramos y ejecutamos la búsqueda en cuadrícula
        grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=tscv,
                                   scoring='neg_mean_squared_error', verbose=0, n_jobs=1)
        grid_search.fit(self.X_train, self.y_train)
        print("Mejores parámetros:", grid_search.best_params_)
        # Almacenamos el mejor modelo encontrado
        self.best_model = grid_search.best_estimator_

        return self.best_model

    def evaluate_model(self):
        predictions = self.best_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        print(f"RMSE: {rmse}")
        return rmse

    def run(self):
        self.prepare_data()
        self.build_and_tune_model()
        return self.evaluate_model()

    def predict_rend(self):
        self.prepare_data()
        self.build_and_tune_model()
        return self.best_model.predict(self.row_month)
