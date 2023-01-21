#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load the dataset
df = pd.read_csv('glass.csv')
df.head()


# # Performing EDA: Exploratory Data Analysis

# In[3]:


#shape of dataframe
df.shape


# In[4]:


#looking for missing values
print(df.isna().sum())


# In[5]:


#checking for data type of elements in each column
df.dtypes


# In[6]:


#looking at statisical summary
df.describe()


# In[7]:


#checking the distinct types in the output column
print(df.groupby('Type')['Type'].count())


# In[8]:


#plotting count for better understanding
sns.countplot(x = df['Type'], color = 'pink')


# In[9]:


#seperating the input and output variables
x = df.iloc[:,:-1].values
x


# In[10]:


y = df.iloc[:,-1].values
y


# In[11]:


#Performing oversampling as the data is imbalanced
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
xo, yo = oversample.fit_resample(x,y)


# In[12]:


#checking oversampling output
y1 = pd.DataFrame(yo)
y1.value_counts()


# In[13]:


#countplot before and after
figure, axes = plt.subplots(1,2,sharex=True, figsize = (10,5))
figure.suptitle('over sampling')
axes[0].set_title('before oversampling')
axes[1].set_title('after oversampling')
sns.countplot(x = y, color = 'pink', ax=axes[0])
sns.countplot(x = yo, color = 'blue', ax=axes[1])


# In[14]:


#splitting test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xo, yo, test_size=0.30,
                                                   random_state = 1)


# In[15]:


#checking the size of train and test data
print ('x_train: ', x_train.shape)
print ('y_train: ', y_train.shape)
print ('x_test: ', x_test.shape)
print ('y_test: ', y_test.shape)


# # KNN MODEL GENERATION

# In[16]:


#normalisation using standard scalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[17]:


#creating knn model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x_train, y_train)


# In[18]:


#predicting output from test values
y_pred = model.predict(x_test)
y_pred


# # PERFORMANCE EVALUATION

# In[19]:


#importing evaluation metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
matrix = confusion_matrix(y_test, y_pred)
matrix


# In[20]:


#displaying confusion matrix
cmd = ConfusionMatrixDisplay(matrix)
cmd.plot()


# In[21]:


score = accuracy_score(y_test, y_pred)
print('The accuracy of this model is: ', score)


# In[22]:


report = classification_report(y_test, y_pred)
print (report)


# # CONCLUSION

# The accuracy of glass prediction using KNN is around 93%

# In[ ]:




