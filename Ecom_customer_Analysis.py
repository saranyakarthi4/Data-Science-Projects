
## Analysis on ecom customers to identify if they spent more time on websites or Apps
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import load_boston
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline


ds1 = pd.read_csv('/Users/skarthi/Documents/Python_Scripts/Data/ecom_customers.csv')
print (ds1.head(2))

sns.heatmap(ds1.corr(),annot=True)

# 1. Joinplot to compare Yearly Ant spend vs Time spend on website
###########sns.jointplot(x='Yearly Amount Spent',y='Time on Website',data=ds1, kind='scatter')

# 2. Create a linear model plot between membership length and Amt spent
###########sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=ds1)

# 3. Fitting the model of  'Yearly amount spent' on all X features
# ds1.columns will give all the columns of dataset
X = ds1[['Avg. Session Length','Time on App','Length of Membership']]
y = ds1['Yearly Amount Spent']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state = 101)
lm= LinearRegression()
lm.fit(X_train,y_train)

print ('Linear line is {}'.format(lm.intercept_))
print ("All the linear line co-efficenits")
co_eff1 = pd.DataFrame(lm.coef_,X_train.columns,columns=['co-efficent'])
print (co_eff1)

# 4. Predicting the model
predictions = lm.predict(X_test)
########plt.scatter(predictions,y_test)
########plt.xlabel('Predicted Y values')
########plt.ylabel('Actual Y values')

# 5. Evaluating the error metrics
print ('mean abs error')
print (metrics.mean_absolute_error(predictions,y_test))
print ('mean squared error')
print (metrics.mean_squared_error(predictions,y_test))

# root mean square error
print (np.sqrt(metrics.mean_squared_error(predictions,y_test)))
print ('explained_variance_score')
# R^2 values, if the R^2 is close to 1 then the it is a good model
print (metrics.explained_variance_score(predictions,y_test))

# also the residual should be a normal distribution
##############sns.distplot(y_test-predictions, bins=50)
