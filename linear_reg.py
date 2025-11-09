import pandas as pd # for data
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


base_dir = Path(__file__).resolve().parent
for _p in [base_dir / 'Ecommerce Customers', base_dir / 'Ecommerce Customers.csv']:
	if _p.exists():
		customers = pd.read_csv(_p) # read csv
		break
else:
	raise FileNotFoundError('Dataset not found next to script') # oops if file not found
print(customers.head()) # check data
customers.info()
print(customers.describe())


sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, alpha=0.5)
plt.show() # plot
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, alpha=0.5)
plt.show() # again show
sns.pairplot(customers, kind='scatter', plot_kws={'alpha':0.4}, diag_kws={'alpha':0.55, 'bins':40})
plt.show() # this one is like all columns
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers, scatter_kws={'alpha':0.3})
plt.show() # line plot thing


from sklearn.model_selection import train_test_split
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # split


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print('Coefficients:', lm.coef_)
print('R^2 score (train):', lm.score(X_train, y_train))
print('R^2 score (test):', lm.score(X_test, y_test))
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
print(cdf)


import statsmodels.api as sm
X_const = sm.add_constant(X_train) # add constant for intercept
model = sm.OLS(y_train, X_const) # make model
model_fit = model.fit() # fit it
print(model_fit.summary()) # summary stats


predictions = lm.predict(X_test)
sns.scatterplot(x=y_test, y=predictions) # compare
plt.xlabel('Actual Yearly Amount Spent')
plt.ylabel('Predictions')
plt.title('Yearly Amount Spent vs. Model Predictions')
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, predictions)))


residuals = y_test - predictions
# use histplot instead of deprecated distplot
sns.histplot(residuals, bins=30, kde=True)
plt.show()
import pylab
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()
