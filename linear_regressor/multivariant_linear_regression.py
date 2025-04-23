import numpy as np
from scipy.special import stdtr, stdtrit
from prettytable import PrettyTable

def check_same_number_of_rows(array1: np.ndarray, array2: np.ndarray):
    """
    Checks if the number of rows in array1 and array2 are the same.

    Raises a ValueError the number of rows are different. 
    """
    if array1.shape[0] != array2.shape[0]:
        raise ValueError(
            f"Arrays have different number of rows: {array1.shape[0]} != {array2.shape[0]}"
        )

def pseudoinverse(M):
        """
        Returns the Moore-Penrose inverse for the matrix M.
        It is computed using the singular value decomposition 
        of M.
        """
        U, s, Vt = np.linalg.svd(M, full_matrices=True)
        Sigma_plus = np.zeros(M.shape).T
        Sigma_plus[:s.shape[0], :s.shape[0]] = np.diag(1 / s)
        M_pinv = np.matmul(
            Vt.T, 
            np.matmul(Sigma_plus, U.T)
        )
        return M_pinv


__all__ = ["LinearRegressor"]

class LinearRegressor(object):
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

    Methods:
        fit(X, y):
            Fits the linear model.

        predict(X):
            Predicts using the linear model.

        score(X, y):
                Returns the coefficient of determination R^2 and the F-statistic.

        get_params():
                Get the parameters (b_0,...,b_p) of the estimator.

        regression_report():
                Returns a report on the statistical analysis 
                of the estimators's coefficients, such as the 
                confidence interval, standard error and p-value.

    """

    def __init__(self, fit_intercept = True):
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

    def fit(self, X, y, sample_weight = None):
        """
        Fits the linear model.

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_predictors).
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

        n_samples = y.shape[0]
        W_sqrt = np.sqrt(self.weight_matrix(n_samples, sample_weight))
        
        if self.fit_intercept:
            X_tilde = np.matmul(
                W_sqrt, 
                np.concatenate((np.ones((n_samples, 1)),X), axis=1)
            )
        else:
            X_tilde = np.matmul(W_sqrt, X)
                      
        X_tilde_pinv = pseudoinverse(X_tilde)
        y_tilde = np.matmul(W_sqrt, y)
        
        b_hat = np.matmul(X_tilde_pinv, y_tilde)

        if self.fit_intercept:
            self.intercept_ = b_hat[0]
            self.coef_ = b_hat[1:]
        else:
            self.coef_ = b_hat[:]

    def predict(self, X): 
        """
        Predicts the values of the dependent variable using 
        the fitted linear model.

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_predictors).
                The predictor's data.

        Returns:
            np.ndarray: array of shape (n_samples,).
        """
        
        n_samples = X.shape[0]
        X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
            
        b_hat = np.append(self.intercept_, self.coef_)
        
        return np.matmul(X_, b_hat)

    def score(self, X, y, sample_weight = None):
        """
        Returns the coefficient of determination R^2 and the F-statistic.
        Given the residual sum of squres (RSS) (y - y_pred)**2.sum and
        the total sum of squares (TSS) (y - y.mean)**2.sum, the 
        coefficient of determination is 1 - RSS/TSS. The F-statistic
        is f*(TSS - RSS)/RSS, and the numerical factor f is given in
        terms of the diagonal sample weight matrix W and the predictors
        X as f = (tr(W)^2 - tr(W)*tr(Q))/(tr(W)*tr(Q) - tr(W^2)), where
        the matrix Q = (X^TWX)^{-1}X^TW^2X. In the case W is the identity
        matrix, f = (n_smaples - n_predictors - 1)/n_predictors

        Keyword arguments:
            X (array-like): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (array-like, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        n_samples = y.shape[0]
        W = self.weight_matrix(n_samples, sample_weight)
        W_sqrt = np.sqrt(W)
        b_ = np.append(self.intercept_, self.coef_) 
        
        X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
        X_tilde = np.matmul(W_sqrt, X_)
        X_tilde_pinv = pseudoinverse(X_tilde)
    
        y_pred = np.matmul(X_, b_)

        res_sum_sqrs = np.matmul((y - y_pred.T), np.matmul(W, y - y_pred))
        y_bar = np.sum(np.matmul(W, y))/np.trace(W)
        tot_sum_sqrs = np.matmul((y - y_bar).T, np.matmul(W, y - y_bar))

        a = np.trace(W) - np.trace(np.matmul(X_tilde_pinv, np.matmul(W, X_tilde)))
        b = np.trace(np.matmul(X_tilde_pinv, np.matmul(W, X_tilde))) - np.trace(W**2)/np.trace(W)
        factor = a/b

        f_stats = factor*(tot_sum_sqrs - res_sum_sqrs)/res_sum_sqrs

        score_table = PrettyTable(["quantity", "value"])
        score_table.add_row(["residual std. error", np.round(np.sqrt(res_sum_sqrs/a), 4)])
        score_table.add_row(["R^2", np.round(1. - res_sum_sqrs/tot_sum_sqrs, 4)])
        score_table.add_row(["F-statistic", np.round(f_stats, 4)])

        return print(score_table)

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
            X (array-like): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (array-like, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        n_samples = y.shape[0]
        W = self.weight_matrix(n_samples, sample_weight)
        W_sqrt = np.sqrt(W)
        
        if self.fit_intercept:
            X_tilde = np.matmul(
                W_sqrt, 
                np.concatenate((np.ones((X.shape[0], 1)),X), axis=1)
            )
            b_ = np.append(self.intercept_, self.coef_)
        else:
            X_tilde = np.matmul(W_sqrt, X)
            b_ = self.coef_[:]

        n_predictors = X_tilde.shape[1]
        y_tilde = np.matmul(W_sqrt, y)

        X_tilde_pinv = pseudoinverse(X_tilde)
        Q = np.matmul(X_tilde.T, X_tilde) 
        M = np.matmul(X_tilde_pinv, np.matmul(W, X_tilde))
        res_sum_sqrd = np.matmul(y_tilde.T, y_tilde) - np.matmul(b_.T, np.matmul(Q, b_))
        variance_hat = res_sum_sqrd/(np.trace(W) - np.trace(M))
        
        conf_level = 0.95
        df = n_samples - n_predictors
        pth_quant_t = np.abs(stdtrit(df, 0.5*(1. - conf_level)))

        res_std_error = np.sqrt(variance_hat*np.diag(np.matmul(X_tilde_pinv, np.matmul(W, X_tilde_pinv.T))))
        p_value = 2.*(1. - stdtr(df, np.abs(b_)/res_std_error))

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

        for i, (coef, res, p) in enumerate(zip(self.coef_, res_std_error[1:], p_value[1:])):
            report_table.add_row(
                [f"coef_{i + 1}", 
                 f"{coef:.4f}", 
                 f"[{coef - pth_quant_t*res:.4f}, {coef + pth_quant_t*res:.4f}]", 
                 f"{res:.4f}", 
                 f"{p:.4f}"
                ]
            )
            
        return print(report_table)
