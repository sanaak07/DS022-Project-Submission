#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Installing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('winequality-red.csv')
data.head()


# In[4]:


data.columns


# In[5]:


#looking at the statistical summary
data.describe()


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data['quality'].value_counts()


# # DATA VISUALISATION
# 
# BIVARIATE ANALYSIS

# In[11]:


#checking variation of fixed acidity in the different qualities
plt.scatter(data['quality'], data['fixed acidity'], color='green')
plt.title('relation of fixed acidity with wine')
plt.xlabel('quality')
plt.ylabel('fixed acidity')
plt.legend()
plt.show()


# In[12]:


#checking the variation of fixed acidity in the different qualities
plt.bar(data['quality'],data['alcohol'], color='maroon')
plt.title('relation of alcohol with wine')
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.legend()
plt.show()


# In[13]:


#composition of citric acid go higher as we go higher in quality
import seaborn as sns
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = data)


# In[14]:


#checking variation in residual sugar
fig = plt.figure(figsize = (10,6))
sns.barplot (x='quality', y = 'residual sugar', data= data)


# In[15]:


#composition of chloride also goes down as we go higher in the quality
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = data)


# In[17]:


#composition of sulphates also goes higher as we go higher in the quality
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = data)


# As we see that like the above 2 items do not have very strong relation to the dependent variable swe have to showcase a correlation plot to check which of the items are more related to the dependent variable and which items are less related to the dependent variables 

# In[20]:


f, ax = plt.subplots(figsize = (10,8))
corr = data.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)


# From the above correlation plot for the given dataset for wine quality prediction, we can easily see which items are related strongly with each other and which items are related weekly with each other. For Example,

# The strongly correlated items are :
# 
# 1.fixed acidity and citric acid. 2.free sulphur dioxide and total sulphor dioxide. 3.fixed acidity and density. 4. alcohol and quality.
# 
# so, from above points there is a clear inference that alcohol is the most important characteristic to determine the quality of wine.

# The weakly correlated items are :
#     
# 1.citric acid and volatile acidity. 2.fixed acidity and ph. 3.density and alcohol.
# 
# These are some relations which do not depend on each other at all.

# In[21]:


sns.pairplot(data)


# # DATA PREPROCESSING

# In[24]:


# Removing Unnecassary columns from the dataset
# As we saw that volatile acidity, total sulphor dioxide, chlorides, density are very less related to the dependent variable  quality so even if we remove these columns the accuracy won't be affected that much.

#data = data.drop(['volatile acidity', 'total sulfur dioxide', 'chlorides', 'density'], axis = 1)

# checking the shape of the dataset
#print(data.shape)


# In[25]:


data.columns


# In[27]:


#converting repsonse to binary
data['quality'] = data['quality'].map({3:'bad', 4:'bad', 5:'bad',6:'good',7:'good',8:'good'})


# In[28]:


#analysing the different values present in the dependent variable (quality column)
data['quality'].value_counts()


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['quality']=le.fit_transform(data['quality'])
data['quality'].value_counts


# In[31]:


sns.countplot(data['quality'])


# In[43]:


#dividing the dataset into dependent and independent variables
x=data.iloc[:,:11]
y=data.iloc[:,11]

#determining the shape of x and y
print(x.shape)
print(y.shape)


# In[39]:


#dividing the dataset in training and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=44)


# In[40]:


#determining shapes of trainng and test sets
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[41]:


#standard scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[ ]:




