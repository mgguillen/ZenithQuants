import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectKBest, f_regression


class PortfolioOptimizer:
    def __init__(self, data=None, p_etfs=None, p_lambda=3, p_W=1, p_Wbar = 0.0125, p_beta = True, p_TB3MS = 0):
        if p_etfs is None:
            self.p_etfs = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLP", "XLY", "XLV", "XLU", "IYR", "VOX"]
        else:
            self.p_etfs = p_etfs

        if data is None:
            print("Por favor ingresa un Set de Datos")
        else:
            self.data = data.loc[:,self.p_etfs].values
        self.weight = np.full(len(self.p_etfs), (1 / len(self.p_etfs)))
        self.Lambda = p_lambda
        self.W = p_W
        self.Wbar = p_Wbar
        self.ind_beta = p_beta
        if p_TB3MS is None:
            self.TB3MS = 0
        else:
            self.TB3MS = p_TB3MS


    def criterion(self, wheight, data):
        """
        -----------------------------------------------------------------------------
        | Output: optimization porfolio criterion                                   |
        -----------------------------------------------------------------------------
        | Inputs: -weight (type ndarray numpy): Wheight for portfolio               |
        |         -data (type ndarray numpy): Returns of stocks                     |
        -----------------------------------------------------------------------------
        """
        #print("MV_criterion-data: ", type(data))
        #print("MV_criterion-data: ", data)
        #portfolio_return, _ = self.data_calc()
        # Compute portfolio returns
        #print("MV_criterion-portfolio_return: ", type(portfolio_return))
        #print("MV_criterion-portfolio_return: ", portfolio_return.head())
        portfolio_return =  np.multiply(data, np.transpose(wheight))
        #print("MV_criterion-portfolio_return: ", type(portfolio_return))
        #print("MV_criterion-portfolio_return: ", portfolio_return.shape)
        portfolio_return_sum = portfolio_return.sum(axis=1)
        #print("portfolio_return_sum: ",portfolio_return_sum)

        # Compute mean and volatility of the portfolio
        mean = np.mean(portfolio_return_sum, axis=0)
        std = np.std(portfolio_return_sum, axis=0)
        std_sortino = np.std(portfolio_return[portfolio_return < 0], axis=0)
        skewness = skew(portfolio_return_sum, 0)
        kurt = kurtosis(portfolio_return_sum, 0)

        # Compute the criterion
        #print("Lambda1: ", self.Lambda)
        ##if self.ind_beta:
        ##    criterion = self.Wbar ** (1 - self.Lambda) / (1 + self.Lambda) + self.Wbar ** (-self.Lambda) \
        ##            * self.W * mean - self.Lambda / 2 * self.Wbar ** (-1 - self.Lambda) * self.W ** 2 * std ** 2 \
        ##            + self.Lambda * (self.Lambda + 1) / (6) * self.Wbar ** (-2 - self.Lambda) * self.W ** 3 * skewness \
        ##            - self.Lambda * (self.Lambda + 1) * (self.Lambda + 2) / (24) * self.Wbar ** (-3 - self.Lambda) * \
        ##            self.W ** 4 * kurt
        ##else:
        criterion = (mean - self.TB3MS)/std_sortino
        #print("Sortino")
        #print(self.TB3MS)
        criterion = -criterion

        return criterion

    def portfolio_optimize(self):
        #portfolio_return = self.data_calc()

        #portfolio_return, _ = self.data_calc()
        #portfolio_return = np.multiply(portfolio_return, np.transpose(self.weight))
        #print("portfolio_optimize-portfolio_return: ", type(self.data))
        #print("portfolio_optimize-portfolio_return: ", self.data.head())
        # Find the number of asset
        n = self.data.shape[1]

        # Initialisation weight value
        x0 = np.array([1/n] * n)

        # Optimization constraints problem
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(abs(x)) - 1})

        # Set the bounds
        Bounds = [(0, 1) for i in range(0, n)]

        # Optimization problem solving
        res_MV = minimize(self.criterion, x0,
                      args=(self.data), method="SLSQP",bounds=Bounds,
                      constraints=cons, options={'disp': True})

        # Result for computations
        #print("Lambda: ", self.Lambda)
        #print(res_MV.x)

        dic_inv = {etf: porcentaje for etf, porcentaje in zip(self.p_etfs, res_MV.x)}

        return dic_inv

        #self.weight = res_MV.x
        #return res_MV.x

    # Compute the cumulative return of the portfolio (CM)
    ##portfolio_return_MV = np.multiply(test_set, np.transpose(X_MV))
    ##portfolio_return_MV = portfolio_return_MV.sum(axis=1)
