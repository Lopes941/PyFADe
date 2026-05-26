import numpy as np
import scipy

from pmdarima import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DEFAULT_AR = 0
DEFAULT_I = 0
DEFAULT_MA = 0

class ArimaFit:

    from .series import DataFrame

    def __init__(self,dataset: DataFrame, column: str | int = 0): 

        if isinstance(column,str):
            self.column = np.where(dataset.dimensions == column)[0]
        else:
            self.column = column

        self.dataset = dataset
        self.test_size = 1
        self.exog_vars = []
        self.AR = DEFAULT_AR
        self.I = DEFAULT_I
        self.MA = DEFAULT_MA
        self.trend = 'c'
        self.with_intercept = True

    def set_order(self, order_AR = DEFAULT_AR, order_I = DEFAULT_I, order_MA = DEFAULT_MA):
        self.AR = order_AR
        self.I = order_I
        self.MA = order_MA

    def set_trend(self, trend):
        self.trend = trend

    def set_intercept(self, with_intercept):
        self.with_intercept = with_intercept

    def set_test_size(self, test_size: int):
        self.test_size = test_size

    def set_exog(self, exog_vars: list | np.ndarray):

        if isinstance(exog_vars[0],int):
            self.exog_vars.extend(exog_vars)
        else:
            loc = np.where([a in exog_vars for a in self.dataset.dimensions])[0]
            self.exog_vars.extend(loc)

    def plot_pacf(self,order = None):

        if order is None:
            order = self.I

        y = self.dataset.dataset[self.column]
        for i in range(order):
            y = np.diff(y)
        plot_pacf(y)

    def plot_acf(self,order = None):

        if order is None:
            order = self.I

        y = self.dataset.dataset[self.column]
        for i in range(order):
            y = np.diff(y)
        plot_acf(y)

    def fit(self):

        mod = ARIMA(order = (self.AR, self.I, self.MA),\
                    trend = self.trend,\
                    with_intercept = self.with_intercept,\
                    out_of_sample_size = self.test_size)
        
         
        
        if len(self.exog_vars)>0:
            self.res = mod.fit(self.dataset.dataset[self.column],X=self.dataset.dataset[self.exog_vars].T,disp=0)
        else:
            self.res = mod.fit(self.dataset.dataset[self.column],disp=0)

        return self.res
    
    def get_prediction(self):

        # ind = self.dataset.index
        # data_name = self.dataset.dimensions[self.column]
        # y = self.dataset.dataset[self.column]

        if len(self.exog_vars)>0:
            exog = self.dataset.dataset[self.exog_vars].T
        else:
            exog = None

        fitted_vals,_ = self.res.predict_in_sample(return_conf_int=True,X=exog)

        return fitted_vals

    def get_residual(self):
        return self.res.resid()


    def plot_fit(self,plot_detailed=False):

        import matplotlib.pyplot as plt
        
        ind = self.dataset.index
        data_name = self.dataset.dimensions[self.column]
        y = self.dataset.dataset[self.column]

        if len(self.exog_vars)>0:
            exog = self.dataset.dataset[self.exog_vars].T
        else:
            exog = None

        if exog is not None:
            fitted, confint = self.res.predict(n_periods=self.test_size,X=exog[-self.test_size:,:], return_conf_int=True)
        else:
            fitted, confint = self.res.predict(n_periods=self.test_size, return_conf_int=True)
        fitted_vals,confint_sample = self.res.predict_in_sample(return_conf_int=True,X=exog)


        n = 40
        plt.plot(ind[n:-self.test_size], y[n:-self.test_size],'b-')
        plt.plot(ind[-self.test_size:], y[-self.test_size:],'r-')

        plt.plot(ind[-self.test_size:],fitted,'g--')


        plt.fill_between(ind[-self.test_size:],confint[-self.test_size:,0],confint[-self.test_size:,1],color='b',alpha=0.1)
        plt.plot(ind[n:-self.test_size],fitted_vals[n:-self.test_size],'g--')
        plt.fill_between(ind[n:-self.test_size],confint_sample[n:-self.test_size,0],confint_sample[n:-self.test_size,1],color='b',alpha=0.1)

        plt.gcf().set_size_inches(10,5)

        plt.xlabel("Time")
        plt.ylabel(f'{data_name}')

        plt.legend(['Train', 'Test', 'Prediction','Confidence interval'])

        if plot_detailed:
            plt.figure()
            plt.plot(ind[-self.test_size:], y[-self.test_size:],'r-')
            plt.plot(ind[-self.test_size:],fitted,'g--')
            plt.fill_between(ind[-self.test_size:],confint[-self.test_size:,0],confint[-self.test_size:,1],color='b',alpha=0.1)
            plt.gcf().set_size_inches(10,5)