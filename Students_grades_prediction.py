#!/usr/bin/env python
# coding: utf-8

# # IMPORTING REQUIRED LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
import warnings 
warnings.filterwarnings('ignore')


# # SETTING VISUALISATION STYLE

# In[2]:


mlp.style.use('seaborn-whitegrid')


# # READING THE DATASET

# In[3]:


grades = pd.read_csv('Grades.csv')
pd.set_option('display.max_columns', None)
grades.head()


# # HELPER FUNCTIONS AND VARIABLES

# In[4]:


# Making seperate lists of courses for each year by filtering out their respective courses.
fe_courses=grades.filter(regex='-1').columns
se_courses=grades.filter(regex='-2').columns
te_courses=grades.filter(regex='-3').columns

# storing lengths of courses for each model

# model 1 having only first year courses
one_year_courses= len(fe_courses)

# model 2 having first two years courses
two_years_courses= len(fe_courses) + len(se_courses)

# model 3 having first three years courses
three_years_courses= len(fe_courses) + len(se_courses) + len(te_courses)


# In[5]:


# returns proportion of not nulll values of a particular row out of the number of courses in a particular model
def valuable_info(not_null_features, no_of_courses):
    return (not_null_features.sum(axis=0))/ no_of_courses

# prints proportions of not null values of a particular row for each model
def print_proportions(row_not_null):
    
    # percentage of not null courses out of total number of first year courses
    not_null_values_fe= row_not_null[(row_not_null==1) & (row_not_null.index.isin(fe_courses))]
    print('FE proportion: ', valuable_info(not_null_values_fe, one_year_courses) * 100)

    # percentage of not null courses out of total number of first year and second year courses 
    not_null_values_se= row_not_null[(row_not_null==1) & (row_not_null.index.isin(fe_courses.append(se_courses)))]
    print('FE + SE proportion: ', valuable_info(not_null_values_se, two_years_courses) * 100)

    # percentage of not null courses out of total number of first year, second year and third year courses
    not_null_values_te= row_not_null[(row_not_null==1) & ~(row_not_null.index.isin(['Seat No.', 'CGPA']))]
    print('FE + SE + TE proportion: ',valuable_info(not_null_values_te, three_years_courses) * 100)


# In[6]:


def scatter_visualization(test_features, test_target, test_predictions, test_size):
    plt.scatter( test_features.index[:test_size],test_target[:test_size], label='Actual')
    plt.scatter( test_features.index[:test_size],test_predictions[:test_size], c='r', alpha=0.5, label='Predicted')
    plt.title('Visualizing actual and predicted values for first 20 test indexes')
    plt.legend(loc= 'best')
    plt.show()

    plt.scatter( test_features.index[-test_size:],test_target[-test_size:], label='Actual')
    plt.scatter( test_features.index[-test_size:],test_predictions[-test_size:], c='r', alpha=0.5, label='Predicted')
    plt.title('Visualizing actual and predicted values for last 20 test indexes')
    plt.legend(loc='best')
    plt.show()


# # ANALYSING DATASET FOR CLEANING OF DATASET

# In[7]:


grades.info()


# * All features besides CGPA are object. 
# * Null values also exist

# # ANALYSING NULL VALUES

# In[8]:


# summing all the null values available in each column
grades.isnull().sum


# Since we don't need 4th year's grade points to predict the CGPA we can drop those features whose course codes start with 4

# In[9]:


# filtering out the fourth year's courses using the '-4' pattern
fourth_year_cols=grades.filter(regex='-4').columns

# dropping the filtered out courses to get courses of only first three years as features
grades.drop(fourth_year_cols, axis=1, inplace=True)


# In[10]:


# checking if the drop was successful
grades.filter(regex='-4')


# In[11]:


# null values left after removal of 4th year courses
grades.isna().sum()


# <b>Analysing null values of each course seperately

# Courses having one null values each

# In[12]:


#Checking CY-105 which has only one null value
grades[grades['CY-105'].isna()]


# From the output it is evident that most of the feature values for this row are null

# In[13]:


grades[grades['HS-105/12'].isna()]


# Common row for null values of CY-105 and HS-105 having Seat No. CS-97045

# In[14]:


cs97045_not_null=grades[grades['Seat No.']=='CS-97045'].notnull().sum()
print(cs97045_not_null[(cs97045_not_null==1) & ~(cs97045_not_null.index.isin(['Seat No.', 'CGPA']))], '\n')
# Only first year's data for CS-97045 has some valuable info.

# Calculating proportion of not null features in cs97045 out of the three models and deciding whether to keep the row or drop it
print('Valuable Information:')
print_proportions(cs97045_not_null)


# CS-97045 provides valuable information for first year courses model only i.e 45.45% since this is also not too much we can either keep the row or drop it.

# In[15]:


#Checking CS-106 which has two null values
grades[grades['CS-106'].isna()]

#One of the null value is for CS-97045 


# In[16]:


cs97282_not_null=grades[grades['Seat No.']=='CS-97282'].notnull().sum()
print(cs97282_not_null[cs97282_not_null==1], '\n')

# cs97282 has null values only for 3rd year

# Calculating proportion of not null features in cs97282 out of the three models and deciding whether to keep the row or drop it
print('Valuable Information:')
print_proportions(cs97282_not_null)


# CS-97282 provides valuable information for all three models so we have decided not to drop it in any of the split datasets.

# In[17]:


grades[grades['MT-111'].isna()]


# In[18]:


cs97566_not_null=grades[grades['Seat No.']=='CS-97566'].notnull().sum()
print(cs97566_not_null[cs97566_not_null==1], '\n')

#Calculating proportion of not null features in cs97566 out of the three models and deciding whether to keep the row or drop it
print('Valuable Information:')
print_proportions(cs97566_not_null)


# CS-97566 provides valuable information for only the first year model.

# In[19]:


grades[grades['EE-119'].isna()]


# In[20]:


cs97283_not_null=grades[grades['Seat No.']=='CS-97283'].notnull().sum()
print(cs97283_not_null[cs97283_not_null==1], '\n')

#Valuable info for all three years

#Calculating proportion of not null features in cs97283 out of the three models and deciding whether to keep the row or drop it
print('Valuable Information:')
print_proportions(cs97283_not_null)


# Overall 9 null values for MT-331 out of which 6 are important

# In[21]:


grades[grades['MT-331'].isna()]


# <B>Valuable information:
# 
# * CS-97143, CS-97144, CS-97138 first year only
# * CS-97061, CS-97092, CS-97289 for first and second

# In[22]:


grades[(grades['CS-317'].isna()) & ~(grades['Seat No.'].isin(['CS-97045', 'CS-97282','CS-97566','CS-97283', 'CS-97143', 'CS-97144', 'CS-97138', 'CS-97061', 'CS-97092', 'CS-97289']))]

# CS-97482 & CS-97544 important for all three years


# # Checking how many null values remain after removing all rows having null values:

# In[23]:


rows_to_drop=grades[grades['Seat No.'].isin(['CS-97045', 'CS-97282','CS-97566','CS-97283', 'CS-97143', 'CS-97144', 'CS-97138', 'CS-97061', 'CS-97092', 'CS-97289', 'CS-97482', 'CS-97544'])]
test_null_dataset= grades.drop(rows_to_drop.index, axis=0)
test_null_dataset.isna().sum()


# <b> Conclusion of null values:
# * rows to drop in first year dataset : CS-97045
# * rows to drop in first + second dataset : CS-97045, CS-97566
# * rows to drop in first + second + third year dataset: CS-97045, * CS-97566, CS-97143, CS-97144, CS-97138

# # Splitting Dataset

# <b> First year dataset

# In[24]:


one_year_df= pd.concat([grades['Seat No.'],grades[fe_courses], grades['CGPA']], axis=1)
one_year_df.head()


# <b> First two years dataset

# In[25]:


two_years_df= pd.concat([grades['Seat No.'],grades[fe_courses], grades[se_courses], grades['CGPA']], axis=1)
two_years_df.head()


# <b> First three years

# In[26]:


three_years= grades
three_years.head()


# # Cleaning Datasets

# <b> Cleaning first year dataset (model 1)

# In[27]:


# Dropping some rows as decided from analysis of dataset
rows_to_drop= one_year_df[one_year_df['Seat No.']=='CS-97045']
one_year_df.drop(rows_to_drop.index, axis=0, inplace=True)

# Filling null values with 0 for rest of the values 
one_year_df.fillna(0, inplace=True)
one_year_df.isna().sum()


# <b> Cleaning first two years dataset (model 2)

# In[28]:


# Dropping some rows as decided from analysis of dataset
rows_to_drop= two_years_df[two_years_df['Seat No.'].isin(['CS-97045', 'CS-97566'])]
two_years_df.drop(rows_to_drop.index, axis=0, inplace=True)

# Filling null values with 0 for rest of the values
two_years_df.fillna(0, inplace=True)
two_years_df.isna().sum()


# <b> Cleaning first three years dataset (model 3)

# In[29]:


# Dropping some rows as decided from analysis of dataset
rows_to_drop= three_years[three_years['Seat No.'].isin(['CS-97045', 'CS-97566', 'CS-97143', 'CS-97144', 'CS-97138'])]
three_years.drop(rows_to_drop.index, axis=0, inplace=True)

# Filling null values with 0 for rest of the values
three_years.fillna(0, inplace=True)
three_years.isna().sum()
     


# # Checking Duplicates

# In[30]:


one_year_df[one_year_df.duplicated('Seat No.')]


# In[31]:


two_years_df[two_years_df.duplicated('Seat No.')]


# In[32]:


three_years[three_years.duplicated('Seat No.')]


# ### No duplicates in any of the datasets (i.e One student has one record only)

# # Data Preprocessing (Feature Encoding)

# <b>Replacing Grades with Grade points

# In[33]:


## creating gp dictionary to be replaced in all three datasets
gp={"A+":4,"A":4,"A-":3.7,"B+":3.4,"B":3.0,"B-":2.7,"C+":2.4,"C":2.0,"C-":1.7,"D+":1.4,"D":1.0,"I":0,"F":0,"WU":0,"W":0}
     


# <b>Replacing in first year dataset

# In[34]:


one_year_df=one_year_df.replace(gp)
one_year_df.head()


# <b>Replacing in first two year dataset

# In[35]:


two_years_df=two_years_df.replace(gp)
two_years_df.head()


# <b>Replacing in first three years dataset

# In[36]:


three_years=three_years.replace(gp)
three_years.head()


# * Checking whether character to numerical encoding was successful or not
# * Successful as all features except for Seat No. are float64 now

# In[37]:


one_year_df.info()


# In[38]:


two_years_df.info()


# In[39]:


three_years.info()


# # Model 1

# In[40]:


one_year_df.drop('Seat No.', axis=1, inplace=True)
features1 = one_year_df.drop('CGPA', axis=1)
target1 = one_year_df['CGPA']


# In[41]:


features1.head()


# In[42]:


target1.head()


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(features1, target1, test_size=0.2, random_state=42)    


# # Support Vector Regression

# In[44]:


svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
model1_score1_train= svr.score(X_train, y_train)
model1_score1_test= svr.score(X_test, y_test)
print('Training Score of  Algorithm: ',model1_score1_train)
print('Testing Score of  Algorithm: ',model1_score1_test)


# In[45]:


scatter_visualization(X_test, y_test, y_pred, 20)


# In[46]:


# Making Function for the algorithm
def svr(X_train,y_train,test):
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(test)
    return y_pred


# # Linear Regression Algorithm

# In[47]:


# Analyzing Linear Regression Algorithm
lr1= LinearRegression()
lr1.fit(X_train, y_train)
y_pred = lr1.predict(X_test)
model1_score2_train= lr1.score(X_train, y_train)
model1_score2_test= lr1.score(X_test, y_test)
print('Training Score of Linear Regression Algorithm: ',model1_score2_train)
print('Testing Score of Linear Regression Algorithm: ',model1_score2_test)


# In[48]:


scatter_visualization(X_test, y_test, y_pred, 20)


# In[49]:


# Making Function for the algorithm
def lr(X_train,y_train,test):
    lr= LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(test)
    return y_pred


# # Model 2

# In[50]:


two_years_df.drop('Seat No.', axis=1, inplace=True)
features2 = two_years_df.drop('CGPA', axis=1)
target2 = two_years_df['CGPA']


# In[51]:


features2.head()


# In[52]:


target2.head()


# In[53]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, target2, test_size=0.2, random_state=42)


# # Support Vector Regression

# In[54]:


svr2 = SVR()
svr2.fit(X_train2, y_train2)
y_pred2 = svr2.predict(X_test2)
model2_score1_train= svr2.score(X_train2, y_train2)
model2_score1_test= svr2.score(X_test2, y_test2)
print('Training Score of  Algorithm: ',model2_score1_train)
print('Testing Score of  Algorithm: ',model2_score1_test)


# In[55]:


scatter_visualization(X_test2, y_test2, y_pred2, 20)


# # Linear Regression Algorithm

# In[56]:


# Analyzing Linear Regression Algorithm
lr2= LinearRegression()
lr2.fit(X_train2, y_train2)
y_pred2 = lr2.predict(X_test2)
model2_score2_train= lr2.score(X_train2, y_train2)
model2_score2_test= lr2.score(X_test2, y_test2)
print('Training Score of Linear Regression Algorithm: ',model2_score2_train)
print('Testing Score of Linear Regression Algorithm: ',model2_score2_test)


# In[57]:


scatter_visualization(X_test2, y_test2, y_pred2, 20)


# # Model 3

# In[58]:


three_years.drop('Seat No.', axis=1, inplace=True)
features3 = three_years.drop('CGPA', axis=1)
target3 = three_years['CGPA']
features3.head()


# In[59]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(features3, target3, test_size=0.2, random_state=42)


# # Support Vector Regression

# In[60]:


svr3 = SVR()
svr3.fit(X_train3, y_train3)
y_pred3 = svr3.predict(X_test3)
model3_score1_train= svr3.score(X_train3, y_train3)
model3_score1_test= svr3.score(X_test3, y_test3)
print('Training Score of  Algorithm: ',model3_score1_train)
print('Testing Score of  Algorithm: ',model3_score1_test)    


# In[61]:


scatter_visualization(X_test3, y_test3, y_pred3, 20)


# # Linear Regression Algorithm

# In[62]:


# Analyzing Linear Regression Algorithm
lr3= LinearRegression()
lr3.fit(X_train3, y_train3)
model3_score2_train= lr3.score(X_train3, y_train3)
model3_score2_test= lr3.score(X_test3, y_test3)
print('Training Score of Linear Regression Algorithm: ',model3_score2_train)
print('Testing Score of Linear Regression Algorithm: ',model3_score2_test)
y_pred3 = lr3.predict(X_test3)


# In[63]:


scatter_visualization(X_test3, y_test3, y_pred3, 20)


# # Comparision of Model Scores

# In[64]:


plt.plot(['model 1','model 2', 'model 3'],[model1_score1_train, model2_score1_train, model3_score1_train])
plt.title('SVR Train')
plt.xlabel('Models')
plt.ylabel('Train Scores')
plt.show()
plt.plot(['model 1','model 2', 'model 3'],[model1_score1_test, model2_score1_test, model3_score1_test])
plt.title('SVR Test')
plt.xlabel('Models')
plt.ylabel('Train Scores')
plt.show()


# In[65]:


plt.plot(['model 1','model 2', 'model 3'],[model1_score2_train, model2_score2_train, model3_score2_train])
plt.title('Linear Regression Train')
plt.xlabel('Models')
plt.ylabel('Train Scores')
plt.show()
plt.plot(['model 1','model 2', 'model 3'],[model1_score2_test, model2_score2_test, model3_score2_test])
plt.title('Linear Regression Test')
plt.xlabel('Models')
plt.ylabel('Train Scores')
plt.show()


# # User Interface

# In[ ]:


while True:
    print('\nSELECT YEAR FROM THE FOLLOWING:\n1. First Year\n2. Second Year\n3. Third Year\n4. Exit')
    year=input('\nPlease enter your year: ')
    if year in ['1','2','3','4']:
       # break
        grs=[]
        if year == '1':
            for courses in fe_courses:
                print("Enter Your grades for",courses) 
                gr=input()
                grs.append(gr.upper())
            testgrades= pd.Series(grs)
            testgrades= testgrades.replace(gp)
            print('\nPrediction by Linear Regression Model\n')
            GPA_lr=lr(X_train,y_train,[testgrades])
            for element in GPA_lr:
                print('\tPredicted CGPA in Fourth year: ',element)
            print('\nPrediction by Support Vector Regression Model\n')
            GPA_svr=svr(X_train,y_train,[testgrades])
            for element in GPA_svr:
                print('\tPredicted CGPA in Fourth year: ',element)

        elif year == '2':
            for courses in features2:
                print("Enter Your grades for",courses) 
                gr=input()
                grs.append(gr.upper())
            testgrades= pd.Series(grs)
            testgrades= testgrades.replace(gp)
            print('\nPrediction by Linear Regression Model\n')
            GPA_lr=lr(X_train2,y_train2,[testgrades])
            for element in GPA_lr:
                print('\tPredicted CGPA in Fourth year: ',element)
            print('\nPrediction by Support Vector Regression Model\n')
            GPA_svr=svr(X_train2,y_train2,[testgrades])
            for element in GPA_svr:
                print('\tPredicted CGPA in Fourth year: ',element)
        elif year== '3':
            for courses in features3:
                print("Enter Your grades for",courses) 
                gr=input()
                grs.append(gr.upper())
            testgrades= pd.Series(grs)
            testgrades= testgrades.replace(gp)
            print('\nPrediction by Linear Regression Model\n')
            GPA_lr=lr(X_train3,y_train3,[testgrades])
            for element in GPA_lr:
                print('\tPredicted CGPA in Fourth year: ',element)
            print('\nPrediction by Support Vector Regression Model\n')
            GPA_svr=svr(X_train3,y_train3,[testgrades])
            for element in GPA_svr:
                print('\tPredicted CGPA in Fourth year: ',element)
        else:
            print('Exit from Interface')
    else:
        print('Invalid Input!\n Use the given options\n')
     


# In[ ]:




