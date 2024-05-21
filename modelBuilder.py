import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import make_scorer,mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV,RandomizedSearchCV,cross_val_score


class ModelBuilder:
    def __init__(self, data, model='LGBMR', split=False, etf='SPY', rand=5,eval_hyper=True, write=False):
        """
        Inicializamos la clase de modelos con los datos .
        :param data: DataFrame con los datos para  entrenar modelos.
        :param split: Booleano para decidir si se realiza una división de los datos
        en el caso de querer evaluar el modelo.
        :param model: seleccionamos el modelo entre 'XGBR', 'LGBMR' y 'RFR'.
        """
        self.data = data
        self.model = model
        self.split = split
        self.row_month = 0
        self.rand = rand
        self.etf = etf
        self.eval_hyper = eval_hyper
        self.write = write

    def prepare_data(self):
        """
        Con esta función preparamos los datos para el modelado, incluyendo la división en características y objetivo
        y si split es True, en train y test.
        """
        self.y = self.data["close"]
        self.X = self.data.drop(["close"], axis=1)
        self.X['date'] = pd.to_datetime(self.X['date'])
        # Buscamos la fecha maxima de los datos
        self.fecha_fin = self.X['date'].max()
        # Calculamos la fecha de inicio para el conjunto de prueba (un año antes de la fecha_fin)
        fecha_inicio_prueba = self.fecha_fin - pd.DateOffset(years=1)
        # Buscamos que los datos sean mayores que esta fecha de inicio
        indice_split = self.X.index[self.X['date'] >= fecha_inicio_prueba].min()
        # Seleccionamos el último registro, que nos servira para la predicción
        self.row_month = self.X.iloc[-1:].drop(['date'], axis=1)
        # Retrasamos todas las caracteristicas, un periodo, asi ponemos la variable objetivo a la misma altura temporal
        # de las caracteristicas dependientes que nos serviran para predecir.
        self.X = self.X.drop(['date'], axis=1).shift(1)
        # Si split es True, separamos los datos en train y test para su evaluación
        if self.split:
            self.X_train = self.X.iloc[:indice_split-1]
            self.y_train = self.y.iloc[:indice_split-1]
            self.X_test = self.X.iloc[indice_split:]
            self.y_test = self.y.iloc[indice_split:]
            if self.X_test.isnull().any().any() or np.isinf(self.X_test).any().any():
                self.X_test.fillna(self.X_test.mean(), inplace=True)
            if self.y_test.isnull().any() or np.isinf(self.y_test).any():
                self.y_test.fillna(self.y_test.mean(), inplace=True)

        else:
            self.X_train = self.X
            self.y_train = self.y
        # Eliminamos valores nulos con la media
        if self.X_train.isnull().any().any() or np.isinf(self.X_train).any().any():
            numeric_cols = self.X_train.select_dtypes(exclude=[np.number])
            self.X_train.fillna(self.X_train.mean(), inplace=True)
        if self.y_train.isnull().any() or np.isinf(self.y_train).any():
            self.y_train.fillna(self.y_train.mean(), inplace=True)


    def build_and_tune(self):
        """
        Entrenamos y afinamos el modelo seleccionado.
        :param model: Método para calcular la importancia ('XGBR', 'LGBMR', 'RFR').
        :return: Devuelve un modelo afinado y entrenado con sus mejores hiperparámetros
        """
        if self.model == 'XGBR':
            return self.build_and_tune_XGBR()
        elif self.model == 'LGBMR':
            return self.build_and_tune_LGBMR()
        elif self.model == 'RFR':
            return self.build_and_tune_RFR()
        else:
            raise ValueError('Unknown model')

    def build_and_tune_XGBR(self):
        #print("builder: ")
        #print(self.X.tail())
        #print(self.X.columns)
        modelo = XGBRegressor(verbosity=0, random_state=42)
        if self.eval_hyper:
            param_grid = {
                'n_estimators':[25, 500],
                'learning_rate':[0.01, 0.05],
                'max_depth': [7,3],
                'subsample': [0.6, 0.8],
                'colsample_bytree':[0.5, 0.7],
                'gamma': [0, 0.5],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 1.5]
            }

        else:
            param_grid = {
                'n_estimators':[25],
                'learning_rate':[0.05],
                'max_depth': [7],
                'subsample': [0.6],
                'colsample_bytree':[0.5],
                'gamma': [0.5],
                'reg_alpha': [0],
                'reg_lambda': [1],
            }
        # Usamos TimeSeriesSplit para la validación cruzada en series temporales
        tscv = TimeSeriesSplit(n_splits=5)
        if self.rand == 0:
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            random_search = RandomizedSearchCV(modelo, param_grid, n_iter=20, scoring=scorer, cv=tscv, random_state=42,
                                           verbose=0)
            random_search.fit(self.X_train, self.y_train)
            parametros = random_search.best_params_
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])
            # Escribir al CSV, añadiendo si el archivo ya existe

            self.best_model = random_search.best_estimator_

        elif self.rand == 1:
            grid_search = GridSearchCV(estimator=modelo,
                                       param_grid=param_grid,
                                       cv=tscv,
                                       scoring='neg_mean_squared_error',
                                       verbose=0,
                                       n_jobs=1,
                                       error_score='raise')
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            parametros = grid_search.best_params_
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=20)
            best_params = study.best_params
            best_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
            best_model.fit(self.X_train, self.y_train)
            self.best_model = best_model
            parametros = best_params
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        if self.eval_hyper and self.write:
            # Escribimos el CSV, añadiendo si el archivo ya existe
            with open('hiperparametros_seleccionados.csv', 'a', newline='') as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        return self.best_model

    def objective(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 2)
        }

        model = XGBRegressor(**param, random_state=42, verbosity=0)

        tscv = TimeSeriesSplit(n_splits=10)
        scores = -cross_val_score(model, self.X_train, self.y_train, cv=tscv, scoring='neg_mean_squared_error')
        rmse = np.mean(np.sqrt(scores))
        return rmse

    def build_and_tune_LGBMR(self):
        modelo = LGBMRegressor(random_state=42, verbosity=-1)
        if self.eval_hyper:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 40, 50],
                'feature_fraction': [0.6, 0.8],
                'bagging_fraction': [0.6, 0.8],
                'max_depth': [10, 15, 20],
                'reg_alpha': [0.1, 0.5],
                'reg_lambda': [1, 1.5]
            }
        else:
            param_grid = {
                'n_estimators':[25],
                'learning_rate':[0.05],
                'max_depth': [7],
                'subsample': [0.6],
                'colsample_bytree':[0.5],
                'gamma': [0.5],
                'reg_alpha': [0],
                'reg_lambda': [1],
            }

        tscv = TimeSeriesSplit(n_splits=5)

        if self.rand == 0:
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            random_search = RandomizedSearchCV(modelo, param_grid, n_iter=20, scoring=scorer, cv=tscv, random_state=42,
                                               verbose=0)
            random_search.fit(self.X_train, self.y_train)
            self.best_model = random_search.best_estimator_
            parametros = random_search.best_params_
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        elif self.rand == 1:
            grid_search = GridSearchCV(modelo, param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=0)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            parametros = grid_search.best_params_
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        else:
            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 50),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.8),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.8),
                    'max_depth': trial.suggest_int('max_depth', 10, 20),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 1.5)
                }
                model = LGBMRegressor(**param, random_state=42, verbosity=-1)
                scores = -cross_val_score(model, self.X_train, self.y_train, cv=tscv, scoring='neg_mean_squared_error')
                rmse = np.mean(np.sqrt(scores))
                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            best_params = study.best_params
            best_model = LGBMRegressor(**best_params, random_state=42, verbosity=-1)
            best_model.fit(self.X_train, self.y_train)
            self.best_model = best_model
            parametros = best_params
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        if self.eval_hyper:
            # Escribimos el CSV, añadiendo si el archivo ya existe
            with open('hiperparametros_seleccionados.csv', 'a', newline='') as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        return self.best_model

    def build_and_tune_RFR(self):
        modelo = RandomForestRegressor(random_state=42)
        if self.eval_hyper:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        else:
            param_grid = {
                'n_estimators': [200],
                'max_depth': [30],
                'min_samples_split': [6],
                'min_samples_leaf': [3],
                'max_features': ['sqrt']
            }

        tscv = TimeSeriesSplit(n_splits=5)

        if self.rand == 0:
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            random_search = RandomizedSearchCV(modelo, param_grid, n_iter=20, scoring=scorer, cv=tscv, random_state=42,
                                               verbose=0)
            random_search.fit(self.X_train, self.y_train)
            self.best_model = random_search.best_estimator_
            parametros = random_search.best_params_
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        elif self.rand == 1:
            grid_search = GridSearchCV(modelo, param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=0)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            parametros = grid_search.best_params_
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])

        else:
            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
                model = RandomForestRegressor(**param, random_state=42)
                scores = -cross_val_score(model, self.X_train, self.y_train, cv=tscv, scoring='neg_mean_squared_error')
                rmse = np.mean(np.sqrt(scores))
                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            best_params = study.best_params
            best_model = RandomForestRegressor(**best_params, random_state=42)
            best_model.fit(self.X_train, self.y_train)
            self.best_model = best_model
            parametros = best_params
            parametros['Modelo'] = self.model
            parametros['Metodo'] = self.rand
            parametros['ETF'] = self.etf
            parametros['date'] = self.fecha_fin
            df = pd.DataFrame([parametros])
        if self.eval_hyper:
            # Escribimos el CSV, añadiendo si el archivo ya existe
            with open('hiperparametros_seleccionados.csv', 'a', newline='') as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        return self.best_model

    def evaluate_model(self):
        """
        Evaluamos el modelo y calculamos el RMSE.
        :return: RMSE del modelo en el conjunto de prueba.
        """
        predictions = self.best_model.predict(self.X_test)
        vrmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        return vrmse

    def predict_rend(self):
         """
         Realizamos una predicción utilizando el mejor modelo ajustado.
         :return: Predicción del modelo para el próximo periodo.
         """
         self.prepare_data()
         self.build_and_tune()
         prediction = self.best_model.predict(self.row_month)

         return prediction[0]

    def run(self):
        """
        Ejecutamos la preparación de datos, construcción, ajuste y evaluación del modelo.
        :return: RMSE de la evaluación del modelo.
        """
        self.prepare_data()
        self.build_and_tune()
        return self.evaluate_model()
