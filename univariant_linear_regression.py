import numpy as np
from scipy.special import stdtrit
from prettytable import PrettyTable

def check_same_number_of_rows(array1: np.ndarray, array2: np.ndarray):
    if array1.shape[0] != array2.shape[0]:
        raise ValueError(
            f"Arrays have different number of rows: {array1.shape[0]} != {array2.shape[0]}"
        )


__all__ = ["SimpleLinearRegressor"]


class SimpleLinearRegressor:
    """
    Ordinary least squares Linear Regressor for one predictor variable.

    SimpleLinearRegressor fits a linear model with coefficients b = (b_0, b_1)
    to minimize the residual sum of squares bewteen the target dataset and
    the linear model b_0 + b_1*x.

    Parameters:
        fit_intercept (bool, default = True):
            If False, it sets the constant term b_0 = 0, i.e. the intercept, 
            and no intercept is used in the minimization.

    Attributes:
        fit_intercept_ (bool): stores the fit_intercept parameter.
        
        coef_ (np.array): array of shape (2,).

    Methods:
        fit(x, y):
            Fits the linear model.

        predict(x):
            Predicts using the linear model.

        score(x, y):
                Returns the coefficient of determination R^2.

        get_params():
                Get the parameters (b_0, b_1) of the estimator.

        regression_report():
                Returns a report on the statistical analysis 
                of the estimators's coefficients, such as the 
                confidence interval, standard error and p-value.

    """

    def __init__(self, 
                 fit_intercept: bool = False
                ):
        """
        Initialises the SimpleLinearRegression with parameters.

        Keyword arguments:
            fit_intercept (bool, default = True):
                If False, it sets the intercept to zero and it 
                is not used in the minimization.
        """

        self.fit_intercept_ = fit_intercept
        self.coef_ = None

    
    def fit(self, x, y):
        """
        Fits the linear model.

        Keyword arguments:
            x (array-like): array of shape (n_samples,).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.

        Returns:
            self (object):
                The fitted estimator.
        """

        n_samples = y.shape[0]

        x_ = np.array(x)
        y_ = np.array(y)

        check_same_number_of_rows(x_, y_)

        x_bar = np.sum(x_)/n_samples
        y_bar = np.sum(y_)/n_samples

        if self.fit_intercept_:
            coef_1 = np.sum((x_ - x_bar)*(y_ - y_bar))/np.sum((x_ - x_bar)**2)
            coef_0 = y_bar - coef_1*x_bar

        else:
            coef_1 = (np.sum((x_ - x_bar)*(y_ - y_bar)) + x_bar*y_bar)/(np.sum((x_ - x_bar)**2) + x_bar**2)
            coef_0 = 0.

        self.coef_ = np.array([coef_0, coef_1])

            
    def predict(self, x):
        """
        Predicts the values of the dependent variable using the fitted linear model.

        Keyword arguments:
            x (array-like): array of shape (n_samples,).
                The predictor's data.

        Returns:
            np.ndarray: array of shape (n_samples,).
        """

        x_ = np.array(x)
        
        return self.coef_[0] + self.coef_[1]*x_

    
    def score(self, x, y):
        """
        Returns the coefficient of determination R^2. The coefficient 
        R^2 is defined as 1 - u/v, where u is the residuals sum of
        squares (y - y_pred)**2.sum and v is the total sum of squares 
        (y - y.mean)**2.sum.

        Keyword arguments:
            x (array-like): array of shape (n_samples,).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.
        """

        x_ = np.array(x)
        y_ = np.array(y)

        check_same_number_of_rows(x_, y_)

        y_bar = np.mean(y_)
        y_pred = self.coef_[0] + self.coef_[1]*x_

        residuals_sum_squares = np.sum((y_ - y_pred)**2)
        total_sum_squares = np.sum((y_ - y_bar)**2)

        return 1. - residuals_sum_squares/total_sum_squares

    
    def get_params(self):

        """
        Get the linear regression coefficients for the estimator.

        Returns:
            C (dict): a dictionary with the coefficient names as keys
            and their correspondent values. 
        """

        return print({'intercept': self.coef_[0], 'coef': self.coef_[1]})

    
    def regression_report(self, x, y):
        """
            Returns a report on the statistical analysis 
            for each of the estimators's coefficients. 
            It gives the confidence interval at 95% confidence
            level.

        Keyword arguments:
            x (array-like): array of shape (n_samples,).
                The predictor's data.

            y (array-like): array of shape (n_samples,).
                The dependent variable's data.
        """

        x_ = np.array(x)
        y_ = np.array(y)

        check_same_number_of_rows(x_, y_)

        n_samples = y.shape[0]
        df = n_samples - 2
        x_bar = np.mean(x_)
        y_bar = np.mean(y_)
        var_error_hat = np.sum((y_ - y_bar)**2 - self.coef_[1]*(x_ - x_bar)**2)/df
        conf_level = 0.95
        p_quant_t = np.abs(stdtrit(df, 0.5*(1. - conf_level)))

        if self.fit_intercept_:
            var_coef_0 = var_error_hat/n_samples + var_error_hat*x_bar**2/np.sum((x_ - x_bar)**2)
            var_coef_1 = var_error_hat/np.sum((x_ - x_bar)**2)
        else:
            var_coef_0 = 0.
            var_coef_1 = var_error_hat/np.sum(x_**2)

        conf_inter_0 = p_quant_t*var_coef_0
        conf_inter_1 = p_quant_t*var_coef_1
        self.conf_inter_ = np.array([conf_inter_0, conf_inter_1])

        print("+----------+---------------------------------+")
        print(f"|          |   confidence interval @ {100*conf_level}%   |")
        print("+----------+---------------------------------+")
        for i, (coef, err) in enumerate(zip(self.coef_, self.conf_inter_)):
            print(f"|  coef_{i}  |   {np.round(coef,10)} ± {np.round(err,10)}   |")
            print("+----------+---------------------------------+")
