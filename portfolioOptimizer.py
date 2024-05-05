import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

np.random.seed(55)


class PortfolioOptimizer:
    def __init__(self, data=None, p_etfs=None, p_TB3MS = 0, p_lambda=3, p_W=1, p_Wbar=0.0125, p_beta=False):
        self.TB3MS = p_TB3MS
        self.p_etfs = p_etfs
        if data is None:
            print("Por favor ingresa un Set de Datos")
            self.data = np.array([])
        elif len(self.p_etfs) > 1:
            self.data = data.loc[:, self.p_etfs].values
        else:
            self.data = np.array([])
        self.weight = np.full(len(self.p_etfs), (1 / len(self.p_etfs))) if self.p_etfs else None
        self.Lambda = p_lambda
        self.W = p_W
        self.Wbar = p_Wbar
        self.ind_beta = p_beta
        self.TB3MS_monthly = (1 + self.TB3MS) ** (1 / 12) - 1

    def criterion(self, weight, data):
        portfolio_return = np.multiply(data, np.transpose(weight))
        portfolio_return_sum = portfolio_return.sum(axis=1)
        mean = np.mean(portfolio_return_sum, axis=0)
        std = np.std(portfolio_return_sum, axis=0)
        skewness = skew(portfolio_return_sum, 0)
        kurt = kurtosis(portfolio_return_sum, 0)
        negative_returns = portfolio_return[portfolio_return < 0]
        std_sortino = np.std(negative_returns) if not negative_returns.size == 0 else np.inf
        if self.ind_beta:
            criterion = self.Wbar ** (1 - self.Lambda) / (1 + self.Lambda) + self.Wbar ** (-self.Lambda) \
                    * self.W * mean - self.Lambda / 2 * self.Wbar ** (-1 - self.Lambda) * self.W ** 2 * std ** 2 \
                    + self.Lambda * (self.Lambda + 1) / (6) * self.Wbar ** (-2 - self.Lambda) * self.W ** 3 * skewness \
                    - self.Lambda * (self.Lambda + 1) * (self.Lambda + 2) / (24) * self.Wbar ** (-3 - self.Lambda) * \
                    self.W ** 4 * kurt
        else:
            criterion = (mean - self.TB3MS_monthly) / (std_sortino + 1e-8)
        criterion = -criterion  # Maximizar criterion
        return criterion

    def portfolio_optimize(self):
        if len(self.p_etfs) > 1:
            n = self.data.shape[1]
            x0 = np.random.rand(n)
            x0 /= np.sum(x0)  # Asegura que la suma es 1
            cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(0.01, 1) for _ in range(n)]
            options = {'maxiter': 1000, 'disp': False}
            res_MV = minimize(self.criterion, x0, args=(self.data), method="SLSQP", bounds=bounds, constraints=cons,
                              options=options)
            dic_inv = {etf: porcentaje for etf, porcentaje in zip(self.p_etfs, res_MV.x)}
            print("Optimized Portfolio Weights:", dic_inv)
            return dic_inv
        else:
            return {self.p_etfs[0]: 1}
