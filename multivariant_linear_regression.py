import numpy as np
from scipy.special import stdtr, stdtrit
from prettytable import PrettyTable

#with gradient descent method and exact line search
class LinearRegressor:
    def __init__(self, fit_intercept = False):
        self.fit_intercept = fit_intercept
        self.eta = 1e-6

    def weight_matrix(self, n_samples, sample_weight):
        
        if sample_weight:
            return np.diag(sample_weight)
        else:
            return np.identity(n_samples)

    def error_sqrd(self, Q, p, c, b):

        return 0.5*np.matmul(b, np.matmul(Q,b)) - np.matmul(p,b) + 0.5*c

    def div_error_sqrd(self, Q, p, b):

        return np.matmul(Q,b) - p
        

    def fit(self, X, y, initial_params = None, sample_weight = None):

        if self.fit_intercept:
            X_ = np.column_stack((np.ones(X.shape[0]),X))
        else:
            X_ = X

        W = self.weight_matrix(y.shape[0], sample_weight)
        Q = np.matmul(X_.T, np.matmul(W, X_))
        p = np.matmul(X_.T, np.matmul(W, y))
        c = np.matmul(y, np.matmul(W, y))

        if initial_params:
            b = initial_params
        else:
            b = np.ones(X_.shape[1])

        div_norm_sqrd = np.inner(
                self.div_error_sqrd(Q, p, b),
                self.div_error_sqrd(Q, p, b)
            )
        while np.sqrt(div_norm_sqrd) > self.eta:
            
            step_size = div_norm_sqrd/(
                np.matmul(self.div_error_sqrd(Q, p, b), 
                         np.matmul(Q, self.div_error_sqrd(Q, p, b))
                         )
            )

            b = b - step_size*self.div_error_sqrd(Q, p, b)
            div_norm_sqrd = np.inner(
                self.div_error_sqrd(Q, p, b),
                self.div_error_sqrd(Q, p, b)
            )
            
        self.coef_ = b

        if self.fit_intercept:
            pass
        else:
            self.coef_ = np.insert(self.coef_,0,0.)

    def predict(self, X): 
        
        X_ = np.column_stack((np.ones(X.shape[0]),X))
        
        return np.matmul(X_, self.coef_)

    def score(self, X, y, sample_weight = None):

        X_ = np.column_stack((np.ones(X.shape[0]),X))
        y_pred = np.matmul(X_,self.coef_)

        W = self.weight_matrix(y.shape[0], sample_weight)

        residuals_squared = np.matmul((y - y_pred), np.matmul(W, y - y_pred))
        y_bar = np.sum(np.matmul(W, y))/np.trace(W)
        variance = np.matmul((y - y_bar), np.matmul(W, y - y_bar))
            
        return 1. - residuals_squared/variance

    def get_params(self):

        params_dict = {}
        for id_, c in enumerate(self.coef_):
            params_dict.update({'coef_' + str(id_): c})        

        return print(params_dict)


    def regression_report(self, X, y, sample_weight = None):

        n_samples = y.shape[0]
        W = self.weight_matrix(n_samples, sample_weight)
        
        X_ = np.concatenate((np.ones((X.shape[0], 1)),X), axis=1)

        n_coefs = X_.shape[1]
            
        Q = np.matmul(X_.T, np.matmul(W, X_))
        Q_inv = np.linalg.inv(Q) 
        M = np.matmul(X_.T, np.matmul(np.matmul(W, W), X_))
        residual_sum_sqrs = np.matmul(y.T, np.matmul(W, y)) - np.matmul(self.coef_.T, np.matmul(Q, self.coef_))
        variance_hat = residual_sum_sqrs/(np.trace(W) - np.trace(np.matmul(Q_inv, M)))
        
        conf_level = 0.95
        df = n_samples - n_coefs
        pth_quant_t = np.abs(stdtrit(df, 0.5*(1. - conf_level)))

        self.res_std_error_ = np.sqrt(variance_hat*np.diag(np.matmul(Q_inv, np.matmul(M, Q_inv))))
        self.p_value_ = 2.*(1. - stdtr(df, np.abs(self.coef_)/self.res_std_error_))

        report_table = PrettyTable(["*****", "coefficient", f"confidence interval @ {100.*conf_level}%", "std. error", "p-value"])

        for i, (coef, res, p) in enumerate(zip(self.coef_, self.res_std_error_, self.p_value_)):
            report_table.add_row([f"coef_{i}", f"{coef:.4f}", f"[{coef - pth_quant_t*res:.4f}, {coef + pth_quant_t*res:.4f}]", f"{res:.4f}", f"{p:.4f}"])
            
        return print(report_table)