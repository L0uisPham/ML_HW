import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf

def load_dataset():
    auto_data = pd.read_csv('data/Auto.csv') 
    return auto_data

def pre_processing(df):
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df.dropna(subset=['horsepower'], inplace=True)
    return df

def linear_regression(df):
    X = sm.add_constant(df['horsepower'])  # Adding a constant
    Y = df['mpg']
    model = sm.OLS(Y, X).fit()
    horsepower_value = 95
    predicted_mpg = model.params['const'] + model.params['horsepower'] * horsepower_value
    plt.scatter(df['horsepower'], df['mpg'], color='blue', label='Actual data', alpha=0.5)
    plt.plot(df['horsepower'], model.predict(X), color='red', label='Regression line')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.title('MPG vs. Horsepower with Regression Line')
    plt.legend()
    plt.show()

def scatter_plot(df):
    sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
    plt.show()

def corr_matrix(df):
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    print(correlation_matrix)

def diag_plots(df):
    X = df.drop(columns=['mpg', 'name'])
    X = sm.add_constant(X)  
    Y = df['mpg']

    multi_regression_model = sm.OLS(Y, X).fit()
  
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].scatter(multi_regression_model.fittedvalues, multi_regression_model.resid)
    axes[0, 0].axhline(y=0, color='grey', linestyle='dashed')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    qqplot(multi_regression_model.resid, line='45', ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')

    axes[1, 0].scatter(multi_regression_model.fittedvalues, np.sqrt(np.abs(multi_regression_model.resid)))
    axes[1, 0].axhline(y=0, color='grey', linestyle='dashed')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('Sqrt of |Residuals|')
    axes[1, 0].set_title('Scale-Location')

    leverage_resid2 = OLSInfluence(multi_regression_model).resid_studentized_internal
    leverages = OLSInfluence(multi_regression_model).hat_matrix_diag
    axes[1, 1].scatter(leverages, leverage_resid2, alpha=0.5)
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_title('Leverage Plot')

    plt.tight_layout()
    plt.show()

def inter_term(df):
    interaction_terms = [
        'cylinders:displacement', 'cylinders:horsepower', 'cylinders:weight', 
        'cylinders:year', 'displacement:horsepower', 'displacement:weight',
        'displacement:year', 'horsepower:weight', 'horsepower:year', 
        'weight:year'
    ]
    formula_with_interactions = 'mpg ~ ' + ' + '.join(interaction_terms)
    model_e = smf.ols(formula=formula_with_interactions, data=df).fit()
    significant_interactions_e = model_e.pvalues[model_e.pvalues < 0.05]
    formula_only_interactions = 'mpg ~ ' + ' + '.join(interaction_terms) + ' -1'
    model_f = smf.ols(formula=formula_only_interactions, data=df).fit()
    significant_interactions_f = model_f.pvalues[model_f.pvalues < 0.05]
    significant_interactions_e, significant_interactions_f
    print(significant_interactions_e)
    print(significant_interactions_f)


if __name__ == '__main__':
    df = load_dataset()
    df = pre_processing(df)
    #linear_regression(df)
    #scatter_plot(df)  
    #corr_matrix(df)
    #diag_plots(df)
    inter_term(df)
