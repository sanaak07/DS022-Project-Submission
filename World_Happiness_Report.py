#!/usr/bin/env python
# coding: utf-8

# # IMPORT REQUIRED LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings ('ignore')


# # READ THE DATASET

# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')


# # CHECK THE DATASET 

# In[3]:


df.head()


# # CHECK THE SHAPE OF DATASET

# In[4]:


df.shape


# In[5]:


df.tail()


# # SEE THE STATISTICAL DATA OF THE DATASET

# In[6]:


df.describe().transpose()


# # CHECK THE COLUMNS OF THE DATASET

# In[7]:


df.columns


# # RENAMING COLUMNS FOR EASE

# In[8]:


df.rename(columns = {'Happiness Rank':'happiness_rank', 'Happiness Score':'happiness_score','Standard Error':'standard_error','Economy (GDP per Capita)':'gdp','Health (Life Expectancy)':'life','Trust (Government Corruption)':'corruption','Dystopia Residual':'dystopia_residual'}, inplace = True)


# In[9]:


df.columns


# In[10]:


df.info()


# In[11]:


# Checking for null values
df.isnull().sum()


# # No null values observed

# In[12]:


#Checking relationship
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df)
plt.savefig('HappinessPairplot.png')
plt.show()


# # REGIONWISE HAPPINESS SCORE

# In[13]:


plt.figure (figsize = (15,12))
sns.boxplot(x='Region', y = 'happiness_score', data=df)
plt.xticks(rotation='vertical')


# In[14]:


col= df.columns


# # GENERATE HEATMAP

# In[15]:


plt.figure(figsize=(18,14))
sns.heatmap(df.corr(), annot=True, linewidth=0.7,linecolor='black', fmt='.2f')


# In[16]:


df.columns


# # DROPPING COLUMNS NOT REQUIRED

# In[17]:


df = df.drop (columns = ['Country','Region','happiness_rank'])


# In[18]:


df


# In[19]:


#cheking the Z-score
from scipy.stats import zscore
z = np.abs(zscore(df))
z


# In[20]:


df_new = df[(z<3).all(axis=1)]
df_new


# In[21]:


#separating x and y
x = df_new.iloc[:,:-1]
y = df_new.iloc[:,-1]


# # MODEL GENERATION

# In[22]:


#building the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


# In[23]:


for i in range(0,1000):
    x_train, x_test,y_train, y_test = train_test_split(x,y, random_state=i, test_size=0.20)
    reg.fit(x_train, y_train)
    print("At random state", i, "the model performs very well")
    print("At random state",i)
    print("Training score is\n ",reg.score(x_train, y_train))
    pred_test= reg.predict(x_test)
    print(f"Predicted value is {pred_test} and Actual value is {y_test.tolist()}\n")
    print('Mean absolute error:- ',mean_absolute_error(y_test, pred_test))
    print('Mean squared error:- ',mean_squared_error(y_test, pred_test))
    print('Root mean squared error:- ',np.sqrt(mean_squared_error(y_test, pred_test)))
    print('r2 Score:- ', r2_score(y_test, pred_test))
    print('*'*80,'\n')


# # ACCURACY SCORE

# In[24]:


reg.score(x_train, y_train) #training score
reg.score(x_test, y_test) #testing score


# # R2 SCORE

# In[25]:


from sklearn.metrics import r2_score
y_pred = reg.predict(x_test)
r2_score(y_test, y_pred)*100

