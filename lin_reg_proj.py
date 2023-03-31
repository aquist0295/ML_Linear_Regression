import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Ecommerce Customers') #This dataset(fictional) is local (<plugin your own dataset>) 
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)

#Explore the data(Data Analysis and Visualization)

#Comparing 2 columns to check correlation between those columns
#sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = df)#correlation btn 'Time on Website' & 'Yearly Amount Spent'
'''
Based on the columns compared:
1. Website users spend 34 to 40 minutes on the website
2. Most purchases occur from 35 to 39 minutes => customers spend more time on the website before purchasing
3. The amount for purchases ranges from $300 to about $750 
4. Most Purchases are made $400 to about $650
NB: There are some outliers as well that fall far below these estimates but not as many to skew the data
'''

#sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data = df)#correlation btn 'Time on App' & 'Yearly Amount Spent'
'''
Based on the columns compared:
1. Website users spend 8 to 15 minutes on the app
2. Most purchases occur from 10 to 14 minutes
3. The amount for purchases ranges from $300 to about $750 
4. Most Purchases are made $400 to about $650
'''

#sns.jointplot(x = 'Time on App', y = 'Length of Membership', data = df, kind = 'hex')#correlation btn 'Time on App' & 'Length of Membership'
'''
Based on the columns compared:
People who spend at least 10 to 14 minutes in the app, usually purchase something
'''

#sns.pairplot(df)

#sns.lmplot(x = 'Length of Membership',y = 'Yearly Amount Spent',data = df)# Best linear correlation
'''
As Length of Membership increases, Yearly Amount Spent increases as well
'''
plt.show()



# Machine Learning
#Splitting data into training and testing data
X = df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 101)

#Training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

#printing the coefficient of the model
print(lm.coef_) #This is also part of evaluating the model

#Predicting Test Data
prediction = lm.predict(X_test)
plt.scatter(y_test, prediction)
plt.xlabel('y_test')
plt.ylabel('Predicted Y')
#plt.show()


#Evaluate the model
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
'''
Mean Abosulute Error(MAE) => absolute difference btn actual data and predicted data divided by the # of errors.
1. if MAE increases => model/metric may be going bad
2. if MAE decreases => model/metric may be going in the right direction i.e in this case linear
3. if results are in consistent model/metric/data might be wrong try another model
'''

print('MSE:', metrics.mean_squared_error(y_test, prediction))
'''
Mean Squared Error(MSE) => absolute difference btn actual data and predicted data divided by the # of errors sqaured.
1. same as above
2. same as above
3. same as above
'''

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
'''
Root Mean Squared Error(RMSE) => (absolute difference btn actual data and predicted data divided by the # of errors sqaured.) square root
1. same as above
2. same as above
3. same as above
'''


#Finding coefficients for all supporting values(X)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
'''
with all other factors fixed what happens if there is a one unit change for each X value
'''
