#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Loading the data
train = pd.read_csv("termdeposit_train.csv")
test = pd.read_csv("termdeposit_test.csv")


# In[3]:


train.columns


# In[4]:


test.columns


# * Hence, 'Subscribed' is the target variable.
# 
# * Checking the data types of the variables

# In[5]:


train.info()


# In[6]:


test.info()


# Checking the shapes of each dataset

# In[7]:


train.shape


# In[8]:


test.shape


# Hence, we can see that we have 17 similar features in the both the dataset and 'Subscribed' is the variable that is to be predicted

# # Data Exploration

# In[9]:


#Printing the first 5 rows of the train dataset
train.head()


# In[10]:


#Printing the first 5 rows of the test dataset
test.head()


# In[11]:


#checking for missing values in train dataset
train.isnull().sum()


# In[12]:


#Checking for missing values in test dataset
test.isnull().sum()


# # Univariate Analysis

# <b>Analysis of 'Subscribed' variable

# In[13]:


#Frequency of 'subscribed'
train['subscribed'].value_counts()


# In[14]:


# Plotting the 'subscribed' frequency
sns.countplot(data=train, x='subscribed')


# In[15]:


#Normalizing the frequency table of 'Subscribed' variable
train['subscribed'].value_counts(normalize=True)


# From the above analysis we can see that only 3,715 people out of 31,647 have subscribed which is roughly 12%.

# <b>Analysing th 'Job' variable

# In[16]:


#Frequency table
train['job'].value_counts()


# In[17]:


# Plotting the job frequency table
sns.set_context('paper')
train['job'].value_counts().plot(kind='bar', figsize=(10,6));


# We can see that most of the clients beloned to blue-collar job and students are least in general as they don't make term deposits in general.

# <b>Analysis of 'marital' status

# In[18]:


train['marital'].value_counts()


# In[19]:


sns.countplot(data=train, x='marital');


# In[20]:


sns.countplot(data=train, x='marital', hue='subscribed');


# # Analyzing the 'age' variable

# In[21]:


sns.distplot(train['age']);


# We can infer that most of the clients fall in the age group between 20-60.

# # Bivariate Analysis

# In[22]:


#job vs subscribed
print(pd.crosstab(train['job'],train['subscribed']))


# In[23]:


job = pd.crosstab(train['job'],train['subscribed'])
job_norm = job.div(job.sum(1).astype(float), axis=0)


# In[24]:


job_norm.plot.bar(stacked=True,figsize=(8,6));


# From the above graph we can infer that students and retired people have higher chances of subscribing to a term deposit, which is surprising as students generally do not subscribe to a term deposit. The possible reason is that the number of students in the dataset is less and comparatively to other job types, more students have subscribed to a term deposit.

# In[25]:


#Marital status vs subscribed
pd.crosstab(train['marital'], train['subscribed'])


# In[26]:


marital = pd.crosstab(train['marital'], train['subscribed'])
marital_norm = marital.div(marital.sum(1).astype(float), axis=0)
marital_norm


# In[27]:


marital_norm.plot.bar(stacked=True, figsize=(10,6));


# From the above analysis we can infer that marital status doesn't have a major impact on the subscription to term deposits.

# In[28]:


#default vs subscription
pd.crosstab(train['default'], train['subscribed'])


# In[29]:


dflt = pd.crosstab(train['default'], train['subscribed'])
dflt_norm = dflt.div(dflt.sum(1).astype(float), axis=0)
dflt_norm


# In[30]:


dflt_norm.plot.bar(stacked=True, figsize=(6,6))


# We can infer that clients having no previous default have slightly higher chances of subscribing to a term loan as compared to the clients who have previous default history.

# In[31]:


# Converting the target variables into 0s and 1s
train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[32]:


train['subscribed']


# In[33]:


#Correlation matrix
tc = train.corr()
tc


# In[34]:


fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')


# We can infer that duration of the call is highly correlated with the target variable. As the duration of the call is more, there are higher chances that the client is showing interest in the term deposit and hence there are higher chances that the client will subscribe to term deposit.

# # Model Building

# In[35]:


target = train['subscribed']
train = train.drop('subscribed', axis=1)


# In[36]:


#generating dummy values on the train dataset
train = pd.get_dummies(train)
train.head()


# Splitting the data into train and validation set such as to validate the results of our model on the validation set. keeping 20% of the dataset as our validation set and the rest as our training set.

# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=12)


# Now our data is ready and it's time to build our model and check its performance. Since it's a classification problem, I'll be using Logistic Regression model for this problem.

# # Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


#creating an object of logistic regression model
lreg = LogisticRegression()


# In[41]:


#fitting the data into the model
lreg.fit(X_train,y_train)


# In[42]:


#Making predictions on the validation set
pred = lreg.predict(X_val)


# Checking the accuracy of our model

# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


#Calculating the accuracy score
accuracy_score(y_val,pred)


# * We got an accuracy score of around 89% on the validation dataset. Logistic regression has a linear decision boundary. What if our data have non linearity? We need a model that can capture this non linearity.
# 
# * Using Decision Tree algorithm to for dealing with non-linearity

# # Decision Tree

# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[46]:


#creating an object of Decision tree
clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# In[47]:


#fitting the model
clf.fit(X_train, y_train)


# In[48]:


#making predictions on the validation set
predict = clf.predict(X_val)
predict


# In[49]:


#Calculating the accuracy
accuracy_score(y_val,predict)


# We got an accuracy of more than 90% on the validation set.
# 
# Let's now make the prediction on test dataset

# In[50]:


test = pd.get_dummies(test)
test.head()


# In[51]:


test_pred = clf.predict(test)
test_pred


# Finally, we will save these predictions into a csv file.

# In[52]:


submissions = pd.DataFrame()


# In[53]:


submissions['ID'] = test['ID']
submissions['subscribed'] = test_pred


# In[54]:


submissions['subscribed']


# Since the target variable is yes or no, we will convert 1 and 0 in the predictions to yes and no respectively.

# In[55]:


submissions['subscribed'].replace(0,'no',inplace=True)
submissions['subscribed'].replace(1,'yes',inplace=True)


# In[56]:


submissions['subscribed']


# In[57]:


submissions.to_csv('submission file.csv', header=True, index=False)

