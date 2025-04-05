import numpy as np
from scipy.special import stdtrit

#one independent variable
class SimpleLinearRegressor:
    def __init__(self, fit_intercept = False):
        self.fit_intercept = fit_intercept

    
    def fit(self, x, y):

        n_samples = y.shape[0]
        x_bar = np.sum(x)/n_samples
        y_bar = np.sum(y)/n_samples

        if self.fit_intercept:
            coef_1 = np.sum((x - x_bar)*(y - y_bar))/np.sum((x - x_bar)**2)
            coef_0 = y_bar - coef_1*x_bar

        else:
            coef_1 = (np.sum((x - x_bar)*(y - y_bar)) + x_bar*y_bar)/(np.sum((x - x_bar)**2) + x_bar**2)
            coef_0 = 0.

        self.coef_ = np.array([coef_0, coef_1])

            
    def predict(self, x):
        
        return self.coef_[0] + self.coef_[1]*x

    
    def score(self, x, y):

        n_samples = y.shape[0]
        y_bar = np.sum(y)/n_samples
        y_pred = self.coef_[0] + self.coef_[1]*x

        residuals_squared = np.sum((y - y_pred)**2)
        variance = np.sum((y - y_bar)**2)

        return 1. - residuals_squared/variance

    
    def get_params(self):

        return print({'coef_0': self.coef_[0], 'coef_1': self.coef_[1]})

    
    def regression_report(self, x, y):

        n_samples = y.shape[0]
        x_bar = np.sum(x)/n_samples
        y_bar = np.sum(y)/n_samples
        var_error_hat = np.sum((y - y_bar)**2 - self.coef_[1]*(x - x_bar)**2)/(n_samples - 2)
        conf_level = 0.99
        df = n_samples - 2
        p_quant_t = np.abs(stdtrit(df, 0.5*(1. - conf_level)))

        if self.fit_intercept:
            var_coef_0 = var_error_hat/n_samples + var_error_hat*x_bar**2/np.sum((x - x_bar)**2)
            var_coef_1 = var_error_hat/np.sum((x - x_bar)**2)
        else:
            var_coef_0 = 0.
            var_coef_1 = var_error_hat/np.sum(x**2)

        conf_inter_0 = p_quant_t*var_coef_0
        conf_inter_1 = p_quant_t*var_coef_1
        self.conf_inter_ = np.array([conf_inter_0, conf_inter_1])

        print("+----------+---------------------------------+")
        print(f"|          |   confidence interval @ {100*conf_level}%   |")
        print("+----------+---------------------------------+")
        for i, (coef, err) in enumerate(zip(self.coef_, self.conf_inter_)):
            print(f"|  coef_{i}  |   {np.round(coef,10)} ± {np.round(err,10)}   |")
            print("+----------+---------------------------------+")