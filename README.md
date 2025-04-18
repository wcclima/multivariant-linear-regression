# Multivariant Linear Regression
A Python implementation of multivariant linear regression with focus on the statistical analysis of the coefficients.

## 1 - Objective

The aim of this project provide a Python-based implementation of the ordinary least squares linear regression method. Its purpose is purely pedagogical and it closely mirrors [scikit-learn's LinearRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). We give some emphasis to the statistical analysis of the linear regression coefficients. In particular, the code computes the standard errors and confidence intervals for the coefficients and their $p$-value against the hypothesis that there is no relation between dependent and preditor variables, as well as the coefficient of determination $R^2$ for the model. These quantities are relevant to assess the accuracy for the estimation of the regression coefficients, the (linear) dependence of the dependent variable $y$ of the predictors $X$ and the overall quality of the fit with the data.

## 2 - Repo organisation

**`linear_regression/`: The linear regression modules**
It contains the modules for univariant and multivariant linear regression. The univariant case is highlighted for pedagogical purposes only. See also Module architecture.

**`notebooks/:` Notebooks demonstrating the modules**
- `LinearRegression.ipynb`: Notebook discussing the basics of linear regression 


## 3 - Module architecture

Description of the `linear_regression` module architecture.

- `linear_regression/__init__.py`
  - Initialises the module.
  - Imports the SimpleLinearRegressor class, for univariant linear regression.
  - Imports the LinearRegressor class, for multivariant linear regression.

- `linear_regression/univariant_linear_regression.py`: defines the SimpleLinearRegressor class with the methods
  - `fit`;
  - `predicts`; 
  - `score`;
  - `get_params`;
  - `regression_report`.

- `linear_regression/multivariant_linear_regression.py`: defines the LinearRegressor class with the same methods as the above.  

## 4 - Features

- The `SimpleLinearRegressor` class:
  - performs an univariant linear regression by minimizing the residual sum of squares;
  - predicts the values $\hat{y}$ of the dependent variable $y$ using the estimates for the coefficients from the predictor values $x$;
  - produces a statistical analysis of the coefficient estimates;
  - has the following methods:
    - `fit` fits the linear model,
    - `predicts` predicts using the linear model,
    - `score` returns the coefficient of determination $R^2$,
    - `get_params` gets the estimation of the regression coefficients,
    - `regression_report` returns a report on the statistical analysis of the estimators's coefficients, with confidence interval at 95% confidence level, standard error and $p$-value against the null hypothesis.

- The `LinearRegressor` class:
  - performs an multivariant linear regression by minimizing the residual sum of squares using the SVD method to obtain the linear system solution;
  - predicts the values $\hat{y}$ of the dependent variable $y$ using the estimates for the coefficients from the predictor values $x$;
  - produces a statistical analysis of the coefficient estimates;
  - has the same methods as the class above.

## 5 - Results

TO DO

## 6 - Bibliography

- [G. Loiola Silva, *Notas de Probabilidade e Estatística* (2024)](https://www.math.tecnico.ulisboa.pt/~gsilva/PE_slides-print.pdf).
- [K. Silva Conceição, *Estatística I*](https://sites.icmc.usp.br/frasson/Estat/material/Estatistica-I-Katiane.pdf).
- G. James, D. Witten, T. Hastie and R. Tibshirani, *An Introduction to Statistical Learning*, Springer (2017).
