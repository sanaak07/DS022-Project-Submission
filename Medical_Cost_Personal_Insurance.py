#!/usr/bin/env python
# coding: utf-8

# Forecasting insurance price for customers using Regression techniques
# 
# The dataset is a collection of medical cost prices for 1338 instances. The objective is to predict the charges for customers based on certain information available about them. Feature set is as follows:
# 
# age: age of primary beneficiary
# sex: insurance contractor gender, female, male
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# children: Number of children covered by health insurance / Number of dependents
# smoker: Smoking status (whether smokes or not)
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# charges: Individual medical costs billed by health insurance
# Predicting the charges will require application of regression algorithms such as Random Forest Regressor and Linear Regression, etc. Before diving in to generation of model, there are some steps necessary to render our data into model understandable and usable format. Also, to understand the type of data we are dealing with, studying its features and statistical analysis of data is required.
# 
# Some of the steps required are -
# 
# Data description - to view and understand how the data looks like, what features exist - their datatypes and values they hold.
# Target variable - The most important aspect of the data. Charges is our target (to predict) and we see how it is distrubuted in the data.
# Data cleaning and pre-processing - finding and handling missing values, checking for valid column names and valid entries for those column, converting data-types of columns in to model acceptable formats and dealing with categorical variables (by generating dummy variables or by updating exisiting features with binary values).
# Data visualization - To generate hidden insights from the data. For example, smokers are charged higher charges than non-smokers.
# Visualization is also required to figure out which features are responsible for changes in the target variable. This is called feature correlation.
# 
# Prepare data, model generation and testing -
# This is the part where Machine learning comes in to picture. Data is divided into training and testing sets. Models are produced by learning training data and finally, their performance is evaluated on testing/unseen data. A good model is capable of accuractely predicting target for unseen instances. A poor model maybe a result of excessive parameter tuning (adjusting parameters to perform well precisely on training data), over-fitting (model learns training data too much and does not understand how to deal with new/unseen feature values) or due to structure of data itself (extremely noisy, messy, highly uncorrelated, unevenly distributed, etc.)

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read data
data = pd.read_csv('medical_cost_insurance.csv')


# In[3]:


#See how top 5 rows of data look like
data.head()


# In[4]:


#See how bottom 5 rows look like
data.tail()


# In[5]:


#Generate statistical summary of the data's numerical features
data.describe()


# # Information from above stats -

# Average age of customers is about 39 years with maximum age of 64 years and they have one child on an average(mean) with minimum of no child and maximum of 5 children. 75% of observations show 51 years of age and 2 children. The charges for insurance on an average(mean) is 13270.42 units with 75% obseravtions close to 16639.91 units.

# In[6]:


#View all column names and their repsective data types
data.info()


# In[7]:


#Check for missing values
print(data.isnull().sum())


# All zeroes mean there are no missing values

# # DATA VISUALISATION

# In[8]:


#Visualize distribution of values for target variable - 'charges'
plt.figure(figsize=(6,6))
plt.hist(data.charges, bins='auto', color='purple')
plt.xlabel('charges->')
plt.title('Distribution of charges values: ')


#     What we know about the target variable?
#     
#     * it is unevenly distributed
#     * most beneficiaries are charges between 1000 to 10,000 units
#     * very few are charged above 50,000
#     * We already know form the statistical data description above that mean is 13270.42 (close to lower limit of target range), which is inclined data towards the left of the ditribution.

# In[9]:


#Generate box-plots to check for outliers and relation of each feature with 'charges' 
cols=['age','children','sex','smoker','region']
for cols in cols:
    plt.figure(figsize=(8,8))
    sns.boxplot(x=data[cols], y=data['charges'])


# # Insights from boxplots generated above -

# * As <b>age</b> increases, insurance cost increases. The plots show an increasing trend (with several small ranges for charges for some ages) in charges starting from around 1000 for age 18-19 to about 10,000 or so for customers with age near 60
#     * This may be due to general medical assumption that younger people are more fit or possess robust immune system.
#     * Another reason could be the types of medical conditions covered by the insurance. If the insurance is designed to cover conditions likely to develop with growing age, charges will be higher for older age groups.
# * <b>Customers with 2 children</b> are charged highest when compared to others. Those with 5 or more children are charged less - This may be due to dominance of group with 2 or 3 children in the entire population.
# * Being a <b>male or female</b> have lesser impact on cost, even though range for males is larger than for females. That means, males are charged higher in several cases than maximum charges for females.
# * The plot shows a clear distribution pattern of high charges for beneficiaries who are <b>smokers</b> and considerably low costs for <b>non-smokers</b>.
# * <b>Region</b> does not show much correlation with charges, though, South-east region have larger range up to about 20,000 in its dsitribution of customer charges. - This could be due to medical costs being higher in the region, some pre-known environmental/physical hazards or because it is a well-developed area with higher costs of living.

# # Converting Categorical Features to Numerical

# In[10]:


clean_data = {'sex': {'male' : 0, 'female' : 1},
             'smoker': {'no':0, 'yes':1},
             'region' : {'northwest':0, 'northeast':1, 'southeast':2, 'southwest':3}
             }
data_copy = data.copy()
data_copy.replace(clean_data, inplace=True)


# In[11]:


data_copy.describe()


# <b>Now we are confirmed taht there are no other values in above pre processed columns, we can proceed with EDA</b>

# In[12]:


plt.figure(figsize=(12,9))
plt.title('Age vs Charges')
sns.barplot(x='age', y='charges',data=data_copy,palette='husl')


# In[13]:


plt.figure(figsize=(10,7))
plt.title('Region vs Charge')
sns.barplot (x='region',y='charges',data=data_copy,palette='Set3')


# In[14]:


plt.figure(figsize=(7,5))
sns.scatterplot(x='bmi',y='charges',hue='sex',data=data_copy, palette='Reds')
plt.title('BMI vs Charge')


# In[15]:


plt.figure(figsize=(10,7))
plt.title('Smoker vs Charge')
sns.barplot(x='smoker',y='charges',data=data_copy,palette='Blues',hue='sex')


# In[16]:


plt.figure(figsize=(10,7))
plt.title('Sex vs Charges')
sns.barplot(x='sex',y='charges',data=data_copy,palette='Set1')


# # Plotting skew and kurtosis

# In[17]:


print('Printing Skewness and Kurtosis for all columns')
print()
for col in list(data_copy.columns):
    print('{0} : Skewness {1:.3f} and  Kurtosis {2:.3f}'.format(col,data_copy[col].skew(),data_copy[col].kurt()))


# In[18]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['age'])
plt.title('Plot for age')
plt.xlabel('Age')
plt.ylabel('Count')


# In[19]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['bmi'])
plt.title('Plot for BMI')
plt.xlabel('BMI')
plt.ylabel('Count')


# In[20]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['charges'])
plt.title('Plot for charges')
plt.xlabel('charges')
plt.ylabel('Count')


# <b>There might be few outliers in Charges but then we cannot say that the value is an outlier as there might be cases in which Charge for medical was very les actually!

# # Prepating data - We can scale BMI and Charges Column before proceeding with Prediction

# In[21]:


from sklearn.preprocessing import StandardScaler
data_pre = data_copy.copy()

tempBmi = data_pre.bmi
tempBmi = tempBmi.values.reshape(-1,1)
data_pre['bmi']=StandardScaler().fit_transform(tempBmi)

tempAge=data_pre.age
tempAge = tempAge.values.reshape(-1,1)
data_pre['age'] = StandardScaler().fit_transform(tempAge)

tempCharges=data_pre.charges
tempCharges = tempCharges.values.reshape(-1,1)
data_pre['charges']=StandardScaler().fit_transform(tempCharges)

data_pre.head()


# In[22]:


X = data_pre.drop('charges', axis=1).values
y = data_pre['charges'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print('Size of X_train : ', X_train.shape)
print('Size of X_test : ', X_test.shape)
print('Size of y_train : ', y_train.shape)
print('Size of y_test : ', y_test.shape)


# # Importing Libraries

# In[23]:


pip install xgboost


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV,GridSearchCV


# # Linear Regression

# In[25]:


get_ipython().run_cell_magic('time', '', 'linear_reg = LinearRegression()\nlinear_reg.fit(X_train, y_train)')


# In[26]:


cv_linear_reg = cross_val_score(estimator = linear_reg, X = X, y = y, cv = 10)

y_pred_linear_reg_train = linear_reg.predict(X_train)
r2_score_linear_reg_train = r2_score(y_train, y_pred_linear_reg_train)

y_pred_linear_reg_test = linear_reg.predict(X_test)
r2_score_linear_reg_test = r2_score(y_test, y_pred_linear_reg_test)

rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_reg_test)))

print('CV Linear Regression : {0:.3f}'.format(cv_linear_reg.mean()))
print('R2_score (train) : {0:.3f}'.format(r2_score_linear_reg_train))
print('R2_score (test) : {0:.3f}'.format(r2_score_linear_reg_test))
print('RMSE : {0:.3f}'.format(rmse_linear))


# In[27]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=False')


# # Support Vector Machine (Regression)

# In[28]:


X_c = data_copy.drop('charges',axis=1).values
y_c = data_copy['charges'].values.reshape(-1,1)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c,y_c,test_size=0.2, random_state=42)

X_train_scaled = StandardScaler().fit_transform(X_train_c)
y_train_scaled = StandardScaler().fit_transform(y_train_c)
X_test_scaled = StandardScaler().fit_transform(X_test_c)
y_test_scaled = StandardScaler().fit_transform(y_test_c)

svr = SVR()
#svr.fit(X_train_scaled, y_train_scaled.ravel())


# In[29]:


parameters =  { 'kernel' : ['rbf', 'sigmoid'],
                 'gamma' : [0.001, 0.01, 0.1, 1, 'scale'],
                 'tol' : [0.0001],
                 'C': [0.001, 0.01, 0.1, 1, 10, 100] }
svr_grid = GridSearchCV(estimator=svr, param_grid=parameters, cv=10, verbose=4, n_jobs=-1)
svr_grid.fit(X_train_scaled, y_train_scaled.ravel())


# In[30]:


svr = SVR(C=10, gamma=0.1, tol=0.0001)
svr.fit(X_train_scaled, y_train_scaled.ravel())
print(svr_grid.best_estimator_)
print(svr_grid.best_score_)


# In[31]:


cv_svr = svr_grid.best_score_

y_pred_svr_train = svr.predict(X_train_scaled)
r2_score_svr_train = r2_score(y_train_scaled, y_pred_svr_train)

y_pred_svr_test = svr.predict(X_test_scaled)
r2_score_svr_test = r2_score(y_test_scaled, y_pred_svr_test)

rmse_svr = (np.sqrt(mean_squared_error(y_test_scaled, y_pred_svr_test)))

print('CV : {0:.3f}'.format(cv_svr.mean()))
print('R2_score (train) : {0:.3f}'.format(r2_score_svr_train))
print('R2 score (test) : {0:.3f}'.format(r2_score_svr_test))
print('RMSE : {0:.3f}'.format(rmse_svr))


# # Ridge Regressor

# In[32]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

steps = [ ('scalar', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Ridge())]

ridge_pipe = Pipeline(steps)


# In[33]:


parameters = { 'model__alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2,1,2,5,10,20,25,35, 43,55,100], 'model__random_state' : [42]}
reg_ridge = GridSearchCV(ridge_pipe, parameters, cv=10)
reg_ridge = reg_ridge.fit(X_train, y_train.ravel())


# In[34]:


reg_ridge.best_estimator_, reg_ridge.best_score_


# In[35]:


ridge = Ridge(alpha=20, random_state=42)
ridge.fit(X_train_scaled, y_train_scaled.ravel())
cv_ridge = reg_ridge.best_score_

y_pred_ridge_train = ridge.predict(X_train_scaled)
r2_score_ridge_train = r2_score(y_train_scaled, y_pred_ridge_train)

y_pred_ridge_test = ridge.predict(X_test_scaled)
r2_score_ridge_test = r2_score(y_test_scaled, y_pred_ridge_test)

rmse_ridge = (np.sqrt(mean_squared_error(y_test_scaled, y_pred_linear_reg_test)))
print('CV : {0:.3f}'.format(cv_ridge.mean()))
print('R2 score (train) : {0:.3f}'.format(r2_score_ridge_train))
print('R2 score (test) : {0:.3f}'.format(r2_score_ridge_test))
print('RMSE : {0:.3f}'.format(rmse_ridge))


# # RandomForest Regressor

# In[36]:


get_ipython().run_cell_magic('time', '', 'reg_rf = RandomForestRegressor()\nparameters = { \'n_estimators\':[600,1000,1200],\n             \'max_features\': ["auto"],\n             \'max_depth\':[40,50,60],\n             \'min_samples_split\': [5,7,9],\n             \'min_samples_leaf\': [7,10,12],\n             \'criterion\': [\'mse\']}\n\nreg_rf_gscv = GridSearchCV(estimator=reg_rf, param_grid=parameters, cv=10, n_jobs=-1)\nreg_rf_gscv = reg_rf_gscv.fit(X_train_scaled, y_train_scaled.ravel())')


# In[37]:


reg_rf_gscv.best_score_, reg_rf_gscv.best_estimator_


# In[38]:


rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7,
                       n_estimators=1200)
rf_reg.fit(X_train_scaled, y_train_scaled.ravel())


# In[39]:


cv_rf = reg_rf_gscv.best_score_

y_pred_rf_train = rf_reg.predict(X_train_scaled)
r2_score_rf_train = r2_score(y_train, y_pred_rf_train)

y_pred_rf_test = rf_reg.predict(X_test_scaled)
r2_score_rf_test = r2_score(y_test_scaled, y_pred_rf_test)

rmse_rf = np.sqrt(mean_squared_error(y_test_scaled, y_pred_rf_test))

print('CV : {0:.3f}'.format(cv_rf.mean()))
print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train))
print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test))
print('RMSE : {0:.3f}'.format(rmse_rf))


# In[40]:


models = [('Linear Regression', rmse_linear, r2_score_linear_reg_train, r2_score_linear_reg_test, cv_linear_reg.mean()),
          ('Ridge Regression', rmse_ridge, r2_score_ridge_train, r2_score_ridge_test, cv_ridge.mean()),
          ('Support Vector Regression', rmse_svr, r2_score_svr_train, r2_score_svr_test, cv_svr.mean()),
          ('Random Forest Regression', rmse_rf, r2_score_rf_train, r2_score_rf_test, cv_rf.mean())   
         ]


# In[41]:


predict = pd.DataFrame(data = models, columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)', 'Cross-Validation'])
predict


# In[42]:


plt.figure(figsize=(12,7))
predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

sns.barplot(x='Cross-Validation', y='Model',data = predict, palette='Reds')
plt.xlabel('Cross Validation Score')
plt.ylabel('Model')
plt.show()


# # Training Data without Scaling for RandomClassifier

# In[43]:


data_copy.head()


# In[44]:


X_ = data_copy.drop('charges',axis=1).values
y_ = data_copy['charges'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_,y_,test_size=0.2, random_state=42)

print('Size of X_train_ : ', X_train_.shape)
print('Size of y_train_ : ', y_train_.shape)
print('Size of X_test_ : ', X_test_.shape)
print('Size of Y_test_ : ', y_test_.shape)


# In[45]:


rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7,
                       n_estimators=1200)
rf_reg.fit(X_train_, y_train_.ravel())


# In[46]:


y_pred_rf_train_ = rf_reg.predict(X_train_)
r2_score_rf_train_ = r2_score(y_train_, y_pred_rf_train_)

y_pred_rf_test_ = rf_reg.predict(X_test_)
r2_score_rf_test_ = r2_score(y_test_, y_pred_rf_test_)

print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train_))
print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test_))


# In[47]:


import pickle

Pkl_Filename = "rf_tuned.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_reg, file)


# In[48]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    rf_tuned_loaded = pickle.load(file)


# In[49]:


rf_tuned_loaded


# In[50]:


pred=rf_tuned_loaded.predict(np.array([20,1,28,0,1,3]).reshape(1,6))[0]


# In[51]:


print('{0:.3f}'.format(pred))

