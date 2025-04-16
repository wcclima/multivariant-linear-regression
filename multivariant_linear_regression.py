import numpy as np
from scipy.special import stdtr, stdtrit
from prettytable import PrettyTable

def check_same_number_of_rows(array1: np.ndarray, array2: np.ndarray):
    if array1.shape[0] != array2.shape[0]:
        raise ValueError(
            f"Arrays have different number of rows: {array1.shape[0]} != {array2.shape[0]}"
        )


__all__ = ["LinearRegressor"]

#with gradient descent method and exact line search
class LinearRegressor:
    """
    Ordinary least squares Linear Regressor.

    LinearRegressor fits a linear model for p predictor variables 
    with coefficients b = (b_1,..., b_p) to minimize the residual 
    sum of squares between the target dataset and the linear model.

    Parameters:
        fit_intercept (bool, default = True):
            If False, it sets the constant term b_0 = 0, i.e. the intercept, 
            and no intercept is used in the minimization.

    Attributes:
        fit_intercept_ (bool): 
            Stores the fit_intercept parameter.
        intercept_ (float): 
            The intercept coefficient.
        coef_ (np.array): array of shape (p_predictors,).
            The predictor's coefficient estimators.
        eta_ (float, default = 1e-6): 
            The precision of the linear system solver.

    Methods:
        fit(X, y):
            Fits the linear model.

        predict(X):
            Predicts using the linear model.

        score(X, y):
                Returns the coefficient of determination R^2.

        get_params():
                Get the parameters (b_0,...,b_p) of the estimator.

        regression_report():
                Returns a report on the statistical analysis 
                of the estimators's coefficients, such as the 
                confidence interval, standard error and p-value.

    """

    def __init__(self, fit_intercept = False):
        """
        Initialises the SimpleLinearRegression with parameters.

        Keyword arguments:
            fit_intercept (bool, default = True):
                If False, it sets the intercept to zero and it 
                is not used in the minimization.
        """

        self.fit_intercept = fit_intercept
        self.intercept_ = 0.
        self.coef_ = None
        self.eta_ = 1e-6

    def weight_matrix(self, n_samples, sample_weight):
        """
        Returns a n_samples x n_samples diagonal matrix
        from the sample_weight array.

        Keyword arguments:
            n_samples (int):
                The number of samples used in the fitting.

            sample_weight (array-like): array of shape (n_samples,) 
                The weight of each sample.
        """
        
        if sample_weight:

            return np.diag(np.array(sample_weight))
        else:
            return np.identity(n_samples)

    def error_sqrd(self, Q, p, c, b):

        return 0.5*np.matmul(b, np.matmul(Q,b)) - np.matmul(p,b) + 0.5*c

    def div_error_sqrd(self, Q, p, b):

        return np.matmul(Q,b) - p
        

    def fit(self, X, y, initial_params = None, sample_weight = None):
        """
        Fits the linear model.

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_features).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (array-like, default = None): array of shape (n_samples,) 
                The weight of each sample.

        Returns:
            self (object):
                The fitted estimator.
        """

        check_same_number_of_rows(X, y)

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
            
        if self.fit_intercept:
            self.intercept_ = b[0]
            self.coef_ = b[1:]
        else:
            self.coef_ = b

    def predict(self, X): 
        """
        Predicts the values of the dependent variable using 
        the fitted linear model.

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_features).
                The predictor's data.

        Returns:
            np.ndarray: array of shape (n_samples,).
        """
        
        X_ = np.column_stack((np.ones(X.shape[0]),X))
        
        return np.matmul(X_, self.coef_)

    def score(self, X, y, sample_weight = None):
        """
        Returns the coefficient of determination R^2. The coefficient 
        R^2 is defined as 1 - u/v, where u is the residuals sum of
        squares (y - y_pred)**2.sum and v is the total sum of squares 
        (y - y.mean)**2.sum.

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_features).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (array-like, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        X_ = np.column_stack((np.ones(X.shape[0]),X))
        y_pred = np.matmul(X_,self.coef_)

        W = self.weight_matrix(y.shape[0], sample_weight)

        residuals_squared = np.matmul((y - y_pred), np.matmul(W, y - y_pred))
        y_bar = np.sum(np.matmul(W, y))/np.trace(W)
        variance = np.matmul((y - y_bar), np.matmul(W, y - y_bar))
            
        return 1. - residuals_squared/variance

    def get_params(self):
        """
        Get the linear regression coefficients for the estimator.

        Returns:
            C (dict): a dictionary with the coefficient names as keys
            and their correspondent values. 
        """

        params_dict = {'intercept': self.intercept_}
        for id_, c in enumerate(self.coef_):
            params_dict.update({'coef_' + str(id_ + 1): c})        

        return print(params_dict)


    def regression_report(self, X, y, sample_weight = None):
        """
            Returns a report on the statistical analysis 
            for each of the estimators's coefficients. 
            It gives the confidence interval at 95% confidence
            level, the standard error and p-value for to reject 
            the hypothesis that the coefficient is null.

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_features).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (array-like, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        n_samples = y.shape[0]
        n_features = X.shape[1]
        W = self.weight_matrix(n_samples, sample_weight)
        
        X_ = np.concatenate((np.ones((X.shape[0], 1)),X), axis=1)
            
        Q = np.matmul(X_.T, np.matmul(W, X_))
        Q_inv = np.linalg.inv(Q) 
        M = np.matmul(X_.T, np.matmul(np.matmul(W, W), X_))
        residual_sum_sqrs = np.matmul(y.T, np.matmul(W, y)) - np.matmul(self.coef_.T, np.matmul(Q, self.coef_))
        variance_hat = residual_sum_sqrs/(np.trace(W) - np.trace(np.matmul(Q_inv, M)))
        
        conf_level = 0.95
        df = n_samples - n_features - 1
        pth_quant_t = np.abs(stdtrit(df, 0.5*(1. - conf_level)))

        res_std_error = np.sqrt(variance_hat*np.diag(np.matmul(Q_inv, np.matmul(M, Q_inv))))
        p_value = 2.*(1. - stdtr(df, np.abs(self.coef_)/res_std_error))

        report_table = PrettyTable(["*****", "coefficient", f"confidence interval @ {100.*conf_level}%", "std. error", "p-value"])
        if self.fit_intercept:
            report_table.add_row(
                [f"intercept", 
                 f"{self.intercept_:.4f}", 
                 f"[{self.intercept_ - pth_quant_t*res_std_error[0]:.4f}, {self.intercept_ + pth_quant_t*res_std_error[0]:.4f}]", 
                 f"{res_std_error[0]:.4f}", 
                 f"{p_value[0]:.4f}"
                ]
            )
        else:
            report_table.add_row([f"intercept", 0., "----", "----", "----"])

        for i, (coef, res, p) in enumerate(zip(self.coef_[1:], res_std_error[1:], p_value[1:])):
            report_table.add_row(
                [f"coef_{i + 1}", 
                 f"{coef:.4f}", 
                 f"[{coef - pth_quant_t*res:.4f}, {coef + pth_quant_t*res:.4f}]", 
                 f"{res:.4f}", 
                 f"{p:.4f}"
                ]
            )
            
        return print(report_table)
