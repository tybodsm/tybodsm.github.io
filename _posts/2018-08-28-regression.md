---
layout: post
title: "A Regression By Any Other Name"
date: 2018-08-28
categories:
  - "Data_Science"
description: The same old song
image: https://picsum.photos/2000/1200?image=244
image-sm: https://picsum.photos/500/300?image=244
---
<!-- 244, 269, 168 -->
<center><i>"Now it's the same old song, but with a different meaning since you been gone"</i><br>
<small>-<b>Holland, Dozier, and Holland </b>
<br>Motown legends and masters of reusing models in different ways</small>  
</center>  
<br><br>

#### TL;DR

As a fundamental model, there are tons of ways to do linear regression in python. Here are a few, from plug-and play to over-engineered 

\*\*_Edited 9/6/18 for typos and additions_ 

## Linear Regression

Linear regression is an old friend of mine. We love to hang out, watch TV, lie to ourselves that we're operating under Gauss-Markov conditions -- it's a grand old time. Linear regression is the workhorse of Economics and has been for long time, prized for prediction ability and interpretability alike. It's also one of the most fundamental machine learning models.

So when I start working on a problem it's one of the first models I go to. It'll give you an idea of feature importance and a baseline prediction. There are various ways to transform it for classification as well, with one of the most popular using the logistic function.

<figure>
    <center>
    <img src="/assets/img/regression_example.jpg"/>
    </center>
</figure>

It's a simple concept: fit a model of the form $$y = X\beta+\alpha$$ to the data by minimizing the sum of squared residuals. Everybody and their brother has written something on the basics of linear regression. But if you're interested in a digestible take on the subject and many of its extensions and applications, I'd suggest [Mostly Harmless Econometrics](http://www.mostlyharmlesseconometrics.com/). 

Anyway, enough blather. Let's go!


## A million ways to regress
Below are excerpts from an ipython notebook, with the full code linked at the bottom. 

### Prep!
You're gonna need a few python packages for this. I recommend Anaconda.

* python 3.6
* numpy
* pandas
* statsmodels
* sklearn
* pytorch

I'll also be using a dataset very near and dear to my heart: [`sysuse auto`](http://www.stata-press.com/data/r9/auto.dta)[^1]. This is a highly regarded and deeply studied dataset in economics.

The basic exercise is to regress automobile price on mpg, headroom, gear_ratio, turn radius, and an indicator for whether the car is foreign.

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_stata('auto.dta')
print(df.head())
```
<div class="tabl">
<table>
  <thead>
    <tr>
      <th>make</th>
      <th>price</th>
      <th>mpg</th>
      <th>rep78</th>
      <th>headroom</th>
      <th>trunk</th>
      <th>weight</th>
      <th>length</th>
      <th>turn</th>
      <th>displacement</th>
      <th>gear_ratio</th>
      <th>foreign</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AMC Concord</td>
      <td>4099</td>
      <td>22</td>
      <td>3.0</td>
      <td>2.5</td>
      <td>11</td>
      <td>2930</td>
      <td>186</td>
      <td>40</td>
      <td>121</td>
      <td>3.58</td>
      <td>Domestic</td>
    </tr>
    <tr>
      <td>AMC Pacer</td>
      <td>4749</td>
      <td>17</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>11</td>
      <td>3350</td>
      <td>173</td>
      <td>40</td>
      <td>258</td>
      <td>2.53</td>
      <td>Domestic</td>
    </tr>
    <tr>
      <td>AMC Spirit</td>
      <td>3799</td>
      <td>22</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>12</td>
      <td>2640</td>
      <td>168</td>
      <td>35</td>
      <td>121</td>
      <td>3.08</td>
      <td>Domestic</td>
    </tr>
    <tr>
      <td>Buick Century</td>
      <td>4816</td>
      <td>20</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>16</td>
      <td>3250</td>
      <td>196</td>
      <td>40</td>
      <td>196</td>
      <td>2.93</td>
      <td>Domestic</td>
    </tr>
    <tr>
      <td>Buick Electra</td>
      <td>7827</td>
      <td>15</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>20</td>
      <td>4080</td>
      <td>222</td>
      <td>43</td>
      <td>350</td>
      <td>2.41</td>
      <td>Domestic</td>
    </tr>
  </tbody>
</table>
</div>


Ok, so we've seen the data. Now we just need a little bit of cleaning and we're ready to go.

```python
y = df.price
X = df[['mpg', 'headroom', 'gear_ratio', 'turn', 'foreign']].assign(const=1)
X.foreign = X.foreign.map({'Domestic': 0, 'Foreign': 1})
print(X.columns)
```
<preout>
<codeout>
Index(['mpg', 'headroom', 'gear_ratio', 'turn', 'foreign', 'const'], dtype='object')
</codeout>
</preout>
<br>


### Statsmodels: Plug and Play

We'll start off easy with statsmodels. 

```python
import statsmodels.api as sm

sm_ols = sm.OLS(y, X).fit()
print(sm_ols.summary())
```

<preout>
<codeout>
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.363
    Model:                            OLS   Adj. R-squared:                  0.316
    Method:                 Least Squares   F-statistic:                     7.743
    Date:                Sun, 19 Aug 2018   Prob (F-statistic):           8.21e-06
    Time:                        19:10:05   Log-Likelihood:                -679.04
    No. Observations:                  74   AIC:                             1370.
    Df Residuals:                      68   BIC:                             1384.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    mpg         -178.4166     77.256     -2.309      0.024    -332.579     -24.254
    headroom    -330.9828    380.210     -0.871      0.387   -1089.679     427.714
    gear_ratio -2665.1207   1045.999     -2.548      0.013   -4752.378    -577.863
    turn         115.1017    113.158      1.017      0.313    -110.702     340.905
    foreign     3577.8473    955.814      3.743      0.000    1670.551    5485.144
    const       1.336e+04   6465.071      2.067      0.043     462.581    2.63e+04
    ==============================================================================
    Omnibus:                       15.566   Durbin-Watson:                   1.428
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               17.603
    Skew:                           1.076   Prob(JB):                     0.000150
    Kurtosis:                       4.039   Cond. No.                     1.04e+03
    ==============================================================================
</codeout>
</preout>
<br>

That was easy! Statsmodels gives you most of the stats info you could want right off the bat, including:

* coefficients
* S.E. on the coefficients
* R-squared
* F-stat
* AIC, BIC
* etc.

The standard errors are particularly nice, telling you about the precision of your estimates. 

This is my preferred method for simple regression work, because getting everything is just too damn simple.

And if you're used to R, I have a special treat for you: [R-style model specification](http://www.statsmodels.org/devel/example_formulas.html). The following code is equivalent to the above. This offers huge flexibility in terms of trying new models on the fly: allowing for lags, categoricals, variable transformations, etc. It even adds the constant for you!

```python
import statsmodels.formula.api as smf

sm_ols = smf.ols('price ~ mpg + headroom + gear_ratio + turn + foreign', data=df).fit()
print(sm_ols.summary())
```

Anyway, remember the coefficients -- you'll be seeing them again (hopefully).


### Scikit-Learn: Pr...

"Wait! Hold on!" you're saying, "Where is the error analysis? The model evaluation? You didn't even do a prediction! Why would you build a model and not pay attention to the error rate?"

### Uh, statsmodels still?

To which I respond with a defiant, "Hey! I don't care." This is not about model evaluation. Every Tom, Dick, and Harry likes talking about prediction accuracy but model interpretation is left for the birds. And for this post we won't need it anyway. So consider this a passive aggressive reaction to prediction-obsession.

<figure>
    <center>
    <img src="/assets/img/ermine.jpg"/>
    <figcaption><i>Let's move on</i></figcaption>
    </center>
</figure>

### Scikit-Learn: Prediction is king

Scikit-learn is also pretty easy. But like most of sklearn, it's focused pretty much solely on prediction. You definitely won't get standard errors on your coefficients, let alone a lot of the other stats traditionally used to describe the fit. If you couldn't tell, I see this as highly undesirable. 

```python
from sklearn import linear_model
from sklearn.utils import resample

sk_ols = linear_model.LinearRegression(fit_intercept=False)
fitted = sk_ols.fit(X, y)
pd.DataFrame({'Variable':X.columns, 'Coef':fitted.coef_})
r2 = fitted.score(X, y)

print('R-squared: {:.3f}'.format(r2))
print(pd.DataFrame({'Variable':X.columns, 'Coef':fitted.coef_}).round(2))
```

<preout>
<codeout>
R-squared: 0.363
</codeout>
</preout>

<div class="tabl">
<table>
  <thead>
    <tr style="text-align: right;">
      <th>Variable</th>
      <th>Coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mpg</td>
      <td>-178.42</td>
    </tr>
    <tr>
      <td>headroom</td>
      <td>-330.98</td>
    </tr>
    <tr>
      <td>gear_ratio</td>
      <td>-2665.12</td>
    </tr>
    <tr>
      <td>turn</td>
      <td>115.10</td>
    </tr>
    <tr>
      <td>foreign</td>
      <td>3577.85</td>
    </tr>
    <tr>
      <td>const</td>
      <td>13363.43</td>
    </tr>
  </tbody>
</table>
</div>
<br>

Great -- the coefficients match! But there are no standard errors. Sklearn can't help here, but we can get by with bootstrapping. Bootstrapping takes repeated subsamples drawn with replacement to estimate what the standard errors are. 

The idea is that the dataset is a subsample drawn from the population. By drawing samples from the data, we simulate this process. So, we draw n samples, run the regression each time, and simply take the standard deviation of the produced estimates. The best part: there's no need to derive the sampling distribution! There are certain cautions to be taken here about selection bias, etc., but I'll leave further research to the reader. [^2]
  


```python
def bootstrap_ols(n, X):
    coef_list = []
    for i in range(n):
        model = linear_model.LinearRegression(fit_intercept=False)
        X_s = resample(X)
        y_s = y.loc[X_s.index]
        coef_list.append(model.fit(X_s, y_s).coef_)
    return coef_list

coef_list = bootstrap_ols(10000, X)

bs_std_err = pd.DataFrame(coef_list).std()
pd.DataFrame({'Variable':X.columns, 'Coef':fitted.coef_, 'Std. Error':bs_std_err}).round(2)
```
<div class="tabl">
<table>
  <thead>
    <tr style="text-align: right;">
      <th>Variable</th>
      <th>Coef</th>
      <th>Std. Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mpg</td>
      <td>-178.42</td>
      <td>103.50</td>
    </tr>
    <tr>
      <td>headroom</td>
      <td>-330.98</td>
      <td>303.99</td>
    </tr>
    <tr>
      <td>gear_ratio</td>
      <td>-2665.12</td>
      <td>1332.82</td>
    </tr>
    <tr>
      <td>turn</td>
      <td>115.10</td>
      <td>123.26</td>
    </tr>
    <tr>
      <td>foreign</td>
      <td>3577.85</td>
      <td>1000.46</td>
    </tr>
    <tr>
      <td>const</td>
      <td>13363.43</td>
      <td>6684.64</td>
    </tr>
  </tbody>
</table>
</div>
<br>

But wait... the standard errors don't match! The issue here is about some of the assumptions made in linear regression. In a nutshell, they rely on asymptotic approximations, so the calculation may not be accurate with such a small sample size.


### Linear Algebra: No school like the old school

So that's all fine and dandy, but it's easy to forget what the heck is actually going on here. 

<figure>
    <center>
    <img src="/assets/img/humility.jpg"/>
    <figcaption><i>I'll tell you the problem with the scientific power that you're using here: it didn't require any discipline to attain it. You know, you read what others had done and you took the next step. You didn't earn the knowledge for yourselves so you don't take any responsibility for it.</i></figcaption>
    </center>
</figure>

The concept of basic linear regression has been around for over a hundred years.[^3] It's come a long way since then, but it has deep roots. A basic introduction to the subject will likely touch on 3 ways to think about the estimation of a linear regression:

* Minimizing the square of the residuals (i.e. calculus and linear algebra)
* Maximum likelihood, assuming a parametric distribution of the errors (usually Gaussian)
* A projection of the dependent (y) vector onto the column space of the regressor matrix (X)

These all basically end up at the same place, though inference depends on the assumptions around the error terms.[^4]

Anyway, the formula you end up with for your coefficients is actually pretty easy! 

$$\hat{\beta} = (X'X)^{-1}X'y$$

The formula for standard errors under a Gaussian assumption is a little worse, with the variance-covariance matrix of the regressors being:

$$E[(\hat{\beta} - \beta)(\hat{\beta} - \beta)'] = \sigma^{2}(X'X)^{-1}$$

where we estimate $$\sigma^{2}$$ with: 

$$\hat{\sigma}^{2} = \frac{e'e}{n-k}$$

Given model:

$$y = X\beta + \epsilon$$

with

$$
\begin{align*}
y     &: \quad \text{dependent vector} \\
X     &: \quad \text{regressor matrix (including a constant column)} \\
\beta &: \quad \text{the coefficient vector} \\
e     &: \quad \text{the error vector (}\hat{y} - y\text{)} \\
k     &: \quad \text{# of regressors} \\
n     &: \quad \text{# of observations}
\end{align*}
$$
<br><br>
or, if you prefer, ($$n-k$$) is the residual degree of freedom


Alright, enough formulas. Let's just put equations blindly into code like a _real_ data scientist


```python
def ols_linalg(X, y):
    X_mat = X.values
    y_mat = y.values
    
    # beta
    XX = np.dot(X_mat.T, X_mat)
    Xy = np.dot(X_mat.T, y_mat)
    inv_XX = np.linalg.inv(XX)
    b = np.dot(inv_XX, Xy)

    y_hat = np.dot(X_mat, b)

    # Std. errors
    e = (y_hat - y_mat)
    sigma2 = np.dot(e.T, e) / (X_mat.shape[0] - X_mat.shape[1])
    var_hat = sigma2 * inv_XX
    std_errs = np.sqrt(var_hat.diagonal())

    return b, std_errs

b_linalg, std_err_linalg = ols_linalg(X, y)

pd.DataFrame({'Variable': X.columns, 'Coef': b_linalg, 'Std. Error': std_err_linalg}).round(2)
```

<div class="tabl">
<table>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Coef</th>
      <th>Std. Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mpg</td>
      <td>-178.42</td>
      <td>77.26</td>
    </tr>
    <tr>
      <td>headroom</td>
      <td>-330.98</td>
      <td>380.21</td>
    </tr>
    <tr>
      <td>gear_ratio</td>
      <td>-2665.12</td>
      <td>1046.00</td>
    </tr>
    <tr>
      <td>turn</td>
      <td>115.10</td>
      <td>113.16</td>
    </tr>
    <tr>
      <td>foreign</td>
      <td>3577.85</td>
      <td>955.81</td>
    </tr>
    <tr>
      <td>const</td>
      <td>13363.43</td>
      <td>6465.07</td>
    </tr>
  </tbody>
</table>
</div>


### Linear Algebra Round 2: Because why not?

We just came from very old mathematical theory, so lets see if we can move this into the 21st century. Just for shits and giggles, we'll soup-up this analysis using the magic of GPUs! Note that if you don't have one, you can simply use pytorch on your CPU.

<figure>
    <center>
    <img src="/assets/img/smdm.jpg"/>
    <figcaption><i>We can rebuild it. We have the technology.</i></figcaption>
    </center>
</figure>

This is pretty straightforward, just moving it from numpy to pytorch.

```python
def ols_linalg_torch(X, y):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    X_mat = torch.from_numpy(X.values).to(device)
    y_mat = torch.from_numpy(y.values).view(-1, 1).double().to(device)
    
    #beta
    XX = torch.mm(X_mat.t(), X_mat)
    Xy = torch.mm(X_mat.t(), y_mat)
    inv_XX = torch.inverse(XX)
    b = torch.mm(inv_XX, Xy)

    y_hat = torch.mm(X_mat, b)

    # Std. errors
    e = torch.sub(y_hat, y_mat)
    sigma2 = torch.mm(e.t(), e) / (X_mat.shape[0] - X_mat.shape[1])
    var_hat = torch.mul(inv_XX, sigma2.expand_as(inv_XX))
    std_errs = torch.sqrt(torch.diag(var_hat))

    return b.to_numpy(), std_errs.to_numpy()   

b_linalg_torch, std_err_linalg_torch = ols_linalg(X, y)

pd.DataFrame({
    'Variable': X.columns, 
    'Coef': b_linalg_torch, 
    'Std. Error': std_err_linalg_torch
}).round(2)

```

<div class="tabl">
<table>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Coef</th>
      <th>Std. Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mpg</td>
      <td>-178.42</td>
      <td>77.26</td>
    </tr>
    <tr>
      <td>headroom</td>
      <td>-330.98</td>
      <td>380.21</td>
    </tr>
    <tr>
      <td>gear_ratio</td>
      <td>-2665.12</td>
      <td>1046.00</td>
    </tr>
    <tr>
      <td>turn</td>
      <td>115.10</td>
      <td>113.16</td>
    </tr>
    <tr>
      <td>foreign</td>
      <td>3577.85</td>
      <td>955.81</td>
    </tr>
    <tr>
      <td>const</td>
      <td>13363.43</td>
      <td>6465.07</td>
    </tr>
  </tbody>
</table>
</div>


Well that was... similar. Wait, I don't know what I was thinking! This is ridiculous. You'd really never get much benefit putting this onto a GPU, unless you have a ton of regressors and matrix inversion gets a much bigger performance improvement than I'm expecting.

It's ok though, I can fix this.


### Neural network: uh, fixed?

Why would we use pytorch and NOT do a neural network? It's silly. 

... ok, so this is just a single neuron, but any network consisting of solely linear activation functions should yield the the same thing -- the weights will just get scattered around. This shouldn't come as any surprise! After all, we're essentially just estimating weights given a linear model as before.


```python
torch.manual_seed(1337)

learn_rate = 0.05
epochs = 75000
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Classic torch net
class Net(torch.nn.Module):
    def __init__(self, k, n_out):
        super(Net, self).__init__()
        self.predict = torch.nn.Linear(k, n_out, bias=True)

    def forward(self, x):
        x = self.predict(x)
        return x
```

That sets up the structure of the network (er, well, neuron), so now we just need to run it. 

**Note:** You need to do some normalization here. Because the regressors are on such different scales, the weights are going to either explode or zero out if you don't do some kind of normalization.[^5] In this case, I only normalize by dividing by the range of each variable, not centering around 0. Why? I eventually want to recover the same weights I had in previous parts, and using an additive term per feature causes a nonlinearity such that I can't back out the OLS weights.  

Anyway, dividing by the range of each variable suffices and is an easy transformation to invert.

```python
X_noconst = X.drop('const', axis=1)
denom = (X_noconst.max() - X_noconst.min())
X_nn = X_noconst/ denom
k = X_nn.shape[1]

X_mat = torch.from_numpy(X_nn.values).double().to(device)
y_mat = torch.from_numpy(y.values).view(-1, 1).double().to(device)

def train_nn(epochs=epochs, learn_rate=learn_rate, verbose=True):
    model = Net(k, 1).to(device).double()     # define the network
    if verbose: print(model)  # net architecture

    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    loss_func = torch.nn.MSELoss()

    for epoch in range(epochs + 1):
        prediction = model(X_mat)     

        loss = loss_func(prediction, y_mat)     

        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()        

        if verbose & ((epoch % 10000) == 0):
            weights = model.predict.weight.data.cpu().numpy().flatten() / denom 

            coefs = np.append(weights, model.predict.bias.data[0].cpu().numpy())

            print(
                f'Epoch: {epoch}', 
                f'Loss: {loss.data[0]}',
                pd.DataFrame({'Variable': X.columns, 'Coef': coefs}),
                '\n',
                sep = '\n'
            )

    weights = model.predict.weight.data.cpu().numpy().flatten() / denom 
    coefs = np.append(weights, model.predict.bias.data[0].cpu().numpy())

    return coefs, model

coefs, model = train_nn(verbose=False)
pd.DataFrame({'Variable': X.columns, 'Coef': coefs}).round(2)
```

<div class="tabl">
<table>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mpg</td>
      <td>-178.42</td>
    </tr>
    <tr>
      <td>headroom</td>
      <td>-330.98</td>
    </tr>
    <tr>
      <td>gear_ratio</td>
      <td>-2665.12</td>
    </tr>
    <tr>
      <td>turn</td>
      <td>115.10</td>
    </tr>
    <tr>
      <td>foreign</td>
      <td>3577.85</td>
    </tr>
    <tr>
      <td>const</td>
      <td>13363.39</td>
    </tr>
  </tbody>
</table>
</div>

Those look familiar!


## Conclusion
So, these are just a few of the ways to run a linear regression in python. It's a powerful, basic model that gives you a lot of tools in one, even if it's not as flashy as some of the more recently developed methods.

Oh! And keep an eye out for extras: I may add a few more bells and whistles.


## References


[^1]: <small>The ["sysuse auto" dataset](http://www.stata-press.com/data/r9/auto.dta) can be found [here](https://www.stata-press.com/data/r9/u.html). Alternatively, I provide [another highly esteemed dataset](https://archive.ics.uci.edu/ml/datasets/iris) which may be easier to use for people with machine learning backgrounds. </small>

[^2]: <small>[More information on the bootstrapping](http://statweb.stanford.edu/~tibs/sta305files/FoxOnBootingRegInR.pdf)</small>

[^3]: <small>[A quick review on the history of the Pearson correlation coefficient and linear regression]( https://ww2.amstat.org/publications/jse/v9n3/stanton.html)</small>

[^4]: <small>[A paper on OLS derivations](https://www.cs.indiana.edu/~predrag/classes/2016fallb365x/ols.pdf) with a quick overview of these topics if you're interested.</small>

[^5]:<small>[A neat post I found](https://theneuralperspective.com/2016/10/27/gradient-topics/) about vanishing gradients and normalization </small>