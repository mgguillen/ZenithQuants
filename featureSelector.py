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


class FeatureSelector:
    def __init__(self, data=None):
        if data is None:
            print("Por favor ingresa un Set de Datos")
        else:
            self.data = data
            #print("datos en featureselector: ",self.data.head())
            #self.data = self.data.dropna()
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')

            #print("datos en featureselector tras dropna: ", self.data.head())
        self.etf = self.data["etf"].unique()
        #print("Datos de y antes de desplazar",self.data["inc_close"])
        #self.y = self.data["inc_close"].shift(1)[1:]
        self.y = self.data["close"]#[1:]
        #print("Datos de y despues de desplazar", self.y)
        #self.X = self.data.drop(['ETF',"inc_close"], axis=1).iloc[0:-1,:]
        self.X = self.data.drop(['date','etf', "close"], axis=1).shift(1)#.iloc[1:, :]
        #print("Datos de X despues de desplazar", self.X.head())
        #print("Datos de X despues de desplazar", self.X.tail())

    def calc_importance_shap(self, modelo):
        explainer = shap.Explainer(modelo)
        shap_values = explainer(self.X)

        # Suma el valor absoluto de los valores SHAP para cada característica
        shap_sum = np.abs(shap_values.values).mean(axis=0)

        # Crea un DataFrame con las características y su importancia SHAP
        features_shap = pd.DataFrame(list(zip(self.X.columns, shap_sum)),
                                     columns=['Feature', 'SHAP Importance'])
        return features_shap

    def select_atributos_shap(self, n_interactions = 10):
        n_interactions = 3  # Número de veces que quieres repetir el proceso
        resultados_shap = pd.DataFrame(index=self.X.columns)

        for i in range(n_interactions):
            # Asignar tratamiento (esto podría variar en cada iteración si lo necesitas)
            tratamiento = np.random.choice([0, 1], size=self.X.shape[0], p=[0.5, 0.5])
            self.X['treatment'] = tratamiento

            ##s_learner = LGBMRegressor()
            s_learner=XGBRegressor(random_state=42)
            #modelo_directo.fit(self.X, self.y)

            # Fit the model
            #print(self.X.columns)
            s_learner.fit(self.X, self.y)

            # Set treatment value to 1
            ##with_treatment = self.X.assign(treatment=1)
            #print(with_treatment.columns)
            # With treatment predictions
            ##with_treatment_predict = s_learner.predict(with_treatment)

            # Set treatment value to 0
            ##without_treatment = self.X.assign(treatment=0)

            # With treatment predictions
            ##without_treatment_predict = s_learner.predict(without_treatment)

            ##ite = with_treatment_predict - without_treatment_predict

            # Save ITE data in a pandas dataframe
            ##ite_df = pd.DataFrame({'ITE': ite, 'with_treatment_predict': with_treatment_predict,
            ##                       'without_treatment_predict': without_treatment_predict})

            # Take a look at the data
            #ite_df.head()

            ##ite_df.hist(column='ITE', bins=50, grid=True, figsize=(12, 8))

            # Calculate ATE
            ##ATE = ite.mean()

            # Print out results
            ##print(f'The average treatment effect (ATE) is {ATE:.2f}')

            # Use XGBRegressor with BaseSRegressor
            ##xgb = BaseSRegressor(XGBRegressor(random_state=42))
            # Estimated ATE, upper bound, and lower bound
            ##tratamiento = self.X.loc[:, "treatment"]
            #print("treatment: ", self.X.tail(), tratamiento)
            ##te, lb, ub = xgb.estimate_ate(self.X, tratamiento, self.y, return_ci=True)
            # Print out results
            ##print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

            # ITE
            ##xgb_ite = xgb.fit_predict(self.X, tratamiento, self.y)
            #print('\nThe all estimated ITEs are:\n', np.matrix(xgb_ite), xgb_ite, xgb_ite.shape)
            # Take a look at the data
            #print('\nThe first five estimated ITEs are:\n', np.matrix(xgb_ite[:10]))

            # ITE with confidence interval
            ##xgb_ite, xgb_ite_lb, xgb_ite_ub = xgb.fit_predict(X=self.X, treatment=self.X[["treatment"]], y=self.y, return_ci=True,
            ##                                                  n_bootstraps=100, bootstrap_size=500)
            # Take a look at the data
            ##print('\nThe first five estimated ITEs are:\n', np.matrix(xgb_ite[:5]))
            ##print('\nThe first five estimated ITE lower bound are:\n', np.matrix(xgb_ite_lb[:5]))
            ##print('\nThe first five estimated ITE upper bound are:\n', np.matrix(xgb_ite_ub[:5]))
            # importancias = xgb.feature_importances_
            ##feature_importances = xgb.get_importance(X=self.X,
            ##                                         tau=xgb_ite,
            ##                                         normalize=True,
            ##                                         method='auto',
            ##                                         features=self.X.columns.tolist())
            ##print(feature_importances)

            # Calcular la importancia de las características
            importancia_shap = self.calc_importance_shap(s_learner)

            # Almacenar los resultados
            resultados_shap[f'iter_{i}'] = importancia_shap.set_index('Feature')['SHAP Importance']

        # Calcular la media de la importancia de SHAP a lo largo de las iteraciones
        resultados_shap['media'] = resultados_shap.mean(axis=1)

        # Características más importantes en promedio
        caracteristicas_importantes = resultados_shap.sort_values('media', ascending=False)

        #print(caracteristicas_importantes)

        return caracteristicas_importantes

    def causal(self):
        tratamiento = np.random.choice([0, 1], size=self.X.shape[0], p=[0.5, 0.5])

        # Asignar esta serie al DataFrame como una nueva columna
        self.X['treatment'] = tratamiento
        # Initiate the light GBM model
        s_learner = LGBMRegressor()

        # Fit the model
        print(self.X.columns)
        s_learner.fit(self.X, self.y)

        # Set treatment value to 1
        with_treatment = self.X.assign(treatment=1)
        print(with_treatment.columns)
        # With treatment predictions
        with_treatment_predict = s_learner.predict(with_treatment)

        # Set treatment value to 0
        without_treatment = self.X.assign(treatment=0)

        # With treatment predictions
        without_treatment_predict = s_learner.predict(without_treatment)

        ite = with_treatment_predict - without_treatment_predict

        # Save ITE data in a pandas dataframe
        ite_df = pd.DataFrame({'ITE': ite, 'with_treatment_predict': with_treatment_predict,
                               'without_treatment_predict': without_treatment_predict})

        # Take a look at the data
        ite_df.head()

        ite_df.hist(column='ITE', bins=50, grid=True, figsize=(12, 8))

        # Calculate ATE
        ATE = ite.mean()

        # Print out results
        print(f'The average treatment effect (ATE) is {ATE:.2f}')

        # Use XGBRegressor with BaseSRegressor
        xgb = BaseSRegressor(XGBRegressor(random_state=42))
        # Estimated ATE, upper bound, and lower bound
        tratamiento = self.X.loc[:,"treatment"]
        print("treatment: ",self.X.tail(),tratamiento)
        te, lb, ub = xgb.estimate_ate(self.X, tratamiento, self.y, return_ci=True)
        # Print out results
        print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

        # ITE
        xgb_ite = xgb.fit_predict(self.X, tratamiento, self.y)
        print('\nThe all estimated ITEs are:\n', np.matrix(xgb_ite),xgb_ite,xgb_ite.shape)
        # Take a look at the data
        print('\nThe first five estimated ITEs are:\n', np.matrix(xgb_ite[:10]))

        # ITE with confidence interval
        ##xgb_ite, xgb_ite_lb, xgb_ite_ub = xgb.fit_predict(X=self.X, treatment=self.X[["treatment"]], y=self.y, return_ci=True,
        ##                                                  n_bootstraps=100, bootstrap_size=500)
        # Take a look at the data
        ##print('\nThe first five estimated ITEs are:\n', np.matrix(xgb_ite[:5]))
        ##print('\nThe first five estimated ITE lower bound are:\n', np.matrix(xgb_ite_lb[:5]))
        ##print('\nThe first five estimated ITE upper bound are:\n', np.matrix(xgb_ite_ub[:5]))
        #importancias = xgb.feature_importances_
        feature_importances=xgb.get_importance(X=self.X,
                                tau=xgb_ite,
                                normalize=True,
                                method='auto',
                                features=self.X.columns.tolist())
        #print(feature_importances)



        # Mostrar las características más importantes
        #print(features_df)
        # Visualization
        xgb.plot_importance(X=self.X, tau=xgb_ite, normalize=True, method='auto', features=self.X.columns.tolist())
        plt.show()

        # Plot shap values
        xgb.plot_shap_values(X=self.X, tau=xgb_ite, features=self.X.columns.tolist())


    def graficar_dispersión(self,columnas_a_seleccionar):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()  # Convierte la matriz de ejes en un array plano para iterar fácilmente.

        for i, var in enumerate(self.X[columnas_a_seleccionar]):
            sns.scatterplot(data=self.data, x=var, y="close", ax=axs[i])
            axs[i].set_title(f'Dispersión entre {var} y precio de {self.etf}')

        plt.tight_layout()
        plt.show()

    def graficar_distribución(self,columnas_a_seleccionar):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()  # Convierte la matriz de ejes en un array plano para iterar fácilmente.

        for i, var in enumerate(self.X[columnas_a_seleccionar]):
            sns.histplot(self.data[var], kde=True, ax=axs[i])
            axs[i].set_title(f'Distribución de {var} para {self.etf}')

        plt.tight_layout()
        plt.show()

    def selectfeature(self):
        #print(X.head())
        #print(self.data.drop(['ETF', "inc_close"], axis=1).iloc[0:-1,:].head())
        #print(X.head())
        #print(X.tail())
        #print(y.head())
        #print(y.tail())

        # Seleccionar las 2 mejores características con f_regression
        selector = SelectKBest(score_func=f_regression, k=10)
        X_new = selector.fit_transform(self.X.drop(['treatment'], axis=1), self.y)
        ##X_new = selector.fit_transform(self.X, self.y)

        #print("Forma original del conjunto de datos:", X.shape)
        #print("Forma del conjunto de datos después de la selección de características:", X_new.shape)
        selected_features = self.data.drop(['ETF',"close"], axis=1).columns[selector.get_support()]

        #print("Características seleccionadas:", selected_features)
        # Identificar las características seleccionadas
        #print("Características seleccionadas:", selector.get_support(indices=True))
        return selected_features
