%matplotlib qt

import numpy as np
import mtrand
import matplotlib.pyplot as plt 

import lime
import lime.lime_tabular
import pandas as pd 
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

from sklearn.linear_model import LinearRegression as sklearn_OLS
from sklearn.datasets import load_boston

##################################
# Setup Data                     #
##################################

def load_data():
    boston_dataset = load_boston()

    cols = [i.lower() for i in boston_dataset.feature_names]
    df = pd.DataFrame(boston_dataset.data, columns=cols)
    df['value'] = boston_dataset.target

    # Pretty sure there are no missing values
    assert ~df.isna().any().any()
    return df


##################################
# Regression and LIME            #
##################################

def regr_LIME(df, features, discretize_continuous=False, observation=5, 
              LIME_model=sklearn_OLS(fit_intercept=True), discretizer='quartile'):

    # The discretizer gets cranky about pandas dataframes, so until that's fixed, this is a simple workaround
    x = df[features].values
    y = df.value.values

    # Model for LIME (sklearn models are easier to deal with for LIME)
    sk_fitted = sklearn_OLS(fit_intercept=True).fit(x, y)

    # Statsmodels, on the otherhand, let's you actually see what the hell happened in the regression
    formula = 'value ~ ' + ' + '.join(features)
    sm_fitted = smf.ols(formula=formula, data=df).fit()
    print(sm_fitted.summary())

    df_out = df.assign(yhat=sm_fitted.predict(df))
    df_out['resid'] = df_out.value - df_out.yhat

    ## LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x,
        mode='regression',
        training_labels=y,
        feature_names=features, 
        class_names=['values'], 
        verbose=True,
        discretize_continuous=discretize_continuous,
        discretizer=discretizer
    )

    # Set a consistent random state for this blog post
    explainer.random_state = mtrand.RandomState(seed=0)

    i = observation
    exp = explainer.explain_instance(x[i], sk_fitted.predict, num_features=7, num_samples=100000, 
                                     model_regressor=LIME_model)

    return df_out, sm_fitted, explainer, exp

def explainer2df(exp):
    coef = (
        pd.DataFrame(exp.as_list(), columns=['var', 'coef'])
        .set_index('var')
        .coef
    )
    intercept = exp.intercept[1]
    return coef, intercept


def denorm_coef(normed_coef, normed_intercept, df_std, df_mean):
    denormed_coef = normed_coef / df_std
    denormed_intercept = normed_intercept - (denormed_coef * df_mean).sum()

    return denormed_coef, denormed_intercept


##################################
# Main section                   #
##################################

# Constants
features = [
    'nox',
    'rm',
    'dis',
    'ptratio',
    'b',
    'lstat',
]
obs=5

#Let's go!

df = load_data()
df_mean = df[features].mean()
df_std = df[features].std()

# Continuous
df_out, sm_fitted, explainer, exp = regr_LIME(df, features, observation=obs)
normed_coef, normed_intercept = explainer2df(exp)
denormed_coef, denormed_intercept = denorm_coef(normed_coef, normed_intercept, df_std, df_mean)

#save plot
exp.as_pyplot_figure()
plt.savefig(f'lime1_obs{obs}_contin.jpg', bbox_inches='tight')

contin_table = pd.concat([
        df[features].loc[obs],
        sm_fitted.params.drop('Intercept'),
        normed_coef,
        denormed_coef,
    ], axis=1)

contin_table.columns = ['value', 'ols_coef', 'normed_coef', 'denormed_coef']

print(contin_table.round(2).to_html())


# Discrete versions
disc = {}
for j in ['quartile', 'decile']:
    d = {}
    d['df_out'], d['sm_fitted'], d['explainer'], d['exp'] = regr_LIME(df, features, discretize_continuous=True,
                                                                      discretizer=j)
    d['coef'], d['intercept'] = explainer2df(d['exp'])
    d['contribution'] = d['coef'].copy()
    d['contribution'].index = d['contribution'].index.str.replace('[^a-z]','')

    disc[j] = d


# Local Prediction info
print(
    'Intercepts:',
    'Quartile-based: ' + str(disc['quartile']['intercept'].round(2)),
    'Decile-based: ' + str(disc['decile']['intercept'].round(2)),
    '',
    'Local predictions:',
    'Quartile-based: ' + str(disc['quartile']['exp'].local_pred[0].round(2)),
    'Decile-based: ' + str(disc['decile']['exp'].local_pred[0].round(2)),
    sep='\n'
)


# Code for saving the plots
for j in ['quartile', 'decile']:
    disc[j]['exp'].as_pyplot_figure()
    plt.savefig(f'lime1_{j}_coefs.jpg', bbox_inches='tight')


## Contributions
df_normalized = (df[features] - df_mean) / df_std
normed_contribution = df_normalized.loc[obs] * normed_coef

ols_contribution = df[features].loc[obs] * denormed_coef

contributions = pd.concat([
        ols_contribution, 
        normed_contribution,
        disc['quartile']['contribution'],
        disc['decile']['contribution'],
    ], axis=1, sort=True)
contributions.loc['intercept'] = [denormed_intercept, normed_intercept, disc['quartile']['intercept'], disc['decile']['intercept']]

contributions.columns = ['ols', 'normed_ols', 'disc_quartile', 'disc_decile']

print(contributions.round(2).to_html())

















