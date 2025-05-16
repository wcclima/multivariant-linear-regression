# Multivariant Linear Regression
A Python implementation of multivariant linear regression with focus on the statistical analysis of the coefficients.

## 1 - Objective

The aim of this project create a Python-based implementation of the ordinary least squares linear regression method. Its purpose is purely pedagogical and it closely mirrors [scikit-learn's LinearRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). We give some emphasis to the statistical analysis of the linear regression coefficients. In particular, the module computes the standard errors and confidence intervals for the coefficients and their $p$-value against the hypothesis that there is no relation between dependent and preditor variables, as well as the coefficient of determination $R^2$ and the $F$-statistic for the model. These quantities are relevant to assess the accuracy for the estimation of the regression coefficients, the (linear) dependence of the dependent variable $y$ of the predictors $X$ and the overall quality of the fit with the data.
c

## 5 - Results

To test and illustrate the `LinearRegressor` class, we generate the artificial data assuming the model 
```math
y = 2.1 + 3.78 x_1 + 7.56 x_2 + 1.29 x_3 + \epsilon,
```
with $\epsilon$ is a normally distributed random error centred in 0. and dispersion 1. In the plot below shows the predicted versus the observed values for the dependent variable $y$. We have used 100 samples for this plot.

![plot](https://github.com/wcclima/multivariant-linear-regression/blob/main/y_hat_vs_y.png)

Using the method `score`, the class returns a statistical analysis of the model by computing the residual standard error, the coefficient of determination $R^2$, the $F$-statistic and the $p$-value. The last two quantities are related to the test of the hypothesis that all the predictors's coefficients are zero versus the alternative that at least one of the coefficients is non-zero.

|       quantity      |  value   |
|:-------------------:|:--------:|
| residual std. error |  1.0029  |
|        $R^2$        |  0.8469  |
|    $F$-statistic    | 177.0768 |
|      $p$-value      |   0.0    |

The method `regression_report` returns a report on the statistical analysis for each of the estimators's coefficients. It gives the confidence interval at 95% confidence level, the standard error and $p$-value for to reject the hypothesis that the coefficient is null.

|           | coefficient | confidence interval @ 95.0% | std. error | p-value |
|----------:|------------:|----------------------------:|-----------:|--------:|
| intercept |    2.3718   |       [1.7778, 2.9657]      |   0.2992   |  0.0000 |
|   coef_1  |    2.8989   |       [2.1736, 3.6241]      |   0.3654   |  0.0000 |
|   coef_2  |    7.5535   |       [6.8090, 8.2980]      |   0.3750   |  0.0000 |
|   coef_3  |    1.6406   |       [0.9870, 2.2941]      |   0.3292   |  0.0000 |


## 6 - Bibliography

- [G. Loiola Silva, *Notas de Probabilidade e Estatística* (2024)](https://www.math.tecnico.ulisboa.pt/~gsilva/PE_slides-print.pdf).
- [K. Silva Conceição, *Estatística I*](https://sites.icmc.usp.br/frasson/Estat/material/Estatistica-I-Katiane.pdf).
- G. James, D. Witten, T. Hastie and R. Tibshirani, *An Introduction to Statistical Learning*, Springer (2017).
- M.N. Magalhães and A.C. Pedroso de Lima, *Noções de Probabilidade e Estatística*, Edusp (2023).
