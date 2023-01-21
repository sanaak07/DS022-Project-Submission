#!/usr/bin/env python
# coding: utf-8

# # 1. IMPORT NECESSARY LIBRARIES

# In[1]:


#data analysis libraries
import numpy as np
import pandas as pd

#visualisation libraries 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# # 2. READ AND EXPLORE THE DATA 
#     

# In[2]:


#import train and test data CSV files
train=pd.read_csv('titanic_train.csv')
test=pd.read_csv('titanic_test.csv')

#take a look on the training data
train.describe(include='all')


# # 3.DATA ANALYSIS

# In[3]:


#get a list of the features within the dataset
print(train.columns)


# In[4]:


#see a sample of the dataset to get an idea of the variables
train.sample(5)


# * NUMERICAL FEATURES: Age(continous), Fare(Continous), SibSp(Discrete), Parch(Discrete)
# * CATEGORICAL FEATURES: Survived, Sex, Embarked, Pclass
# * ALPHANUMERICAL FEATURES: Ticket, Cabin

# In[5]:


train.dtypes


# In[6]:


#see a summary of the training dataset 
train.describe(include='all')


# Some observations:
# * there are a total of 891 passengers in our training set
# * The age feature is missing approx 19.8% of its values. I'm guessing that the age feature is pretty much important to survival, so we should probably attempt to fill these gaps;
# * The cabin feature is missing approx.77.1%since so much of the feature is missing it would be hard to fill in these missing values. We'll probably drop these values from our dataset
# * The embarked feature is missing 0.22% of its values, which should be relatively harmless
# 

# In[7]:


#check for nay other unusable values
print(pd.isnull(train).sum)


# We can see that except for the above mentioned missing values, no NaN values exist

# Some Predictions:
#     
# * Sex: Females are more likely to survive
# * SibSp: People travelling alone are more likely to survive 
# * Age: Young children are more likely to survive
# * Pclass:People of higher socioeconomic class are more likely to survive

# # 4. DATA VISUALISATION

# SEX FEATURE

# In[8]:


#draw a bar plot of survival by sex
sns.barplot(x='Sex',y='Survived',data=train)

#print percentages of females vs males taht survive
print('Percentage of females survived', train['Survived'][train['Sex']=='female'].value_counts(normalize=True)[1]*100)

print('Percentage of males survived', train['Survived'][train['Sex']=='male'].value_counts(normalize=True)[1]*100)


# As predicted, females have a much higher chance of survival than males. The sex feature is essential in our predictions

# PCLASS FEATURE

# In[9]:


#draw a bar plot of survival by Pclass
sns.barplot(x='Pclass', y='Survived',data=train)

#print percentage of people by Pclass that survived
print('Percentage of Pclass=1 who survived: ', train['Survived'][train['Pclass']==1].value_counts(normalize=True[1]*100))

print('Percentage of Pclass=2 who survived: ', train['Survived'][train['Pclass']==2].value_counts(normalize=True[1]*100))

print('Percentage of Pclass=3 who survived: ', train['Survived'][train['Pclass']==3].value_counts(normalize=True[1]*100))


# As predicted, people with higher socioeconomic class had a higher rate of survival(62.9% vs 47.3% vs.24.2%)

# SIBSP FEATURE

# In[10]:


#draw a bar plot for SibSp vs.survival
sns.barplot(x='Parch',y='Survived', data=train)

#I won't be printing individual percent values for all of these
print ('Percentage of SipSp = 0 who survived: ', train['Survived'][train['SibSp']==0].value_counts(normalize=True)[1]*100)

print ('Percentage of SipSp = 1 who survived: ', train['Survived'][train['SibSp']==1].value_counts(normalize=True)[1]*100)

print ('Percentage of SipSp = 2 who survived: ', train['Survived'][train['SibSp']==3].value_counts(normalize=True)[1]*100)


# In general, it's clear that people with more siblings or spouses aboard were less likely to survive. However, contrary to expectations, people with no siblings or spouses were less to likely to survive than those with one or two. (34.5% vs 53.4% vs. 46.4%)

# PARCH FEATURE

# In[11]:


#draw a bar plot for Parch vs.Survival
sns.barplot(x='Parch',y='Survived', data=train)
plt.show()


# People with less than four parents or children aboard are more likely to survive than those with four or more. Again, people traveling alone are less likely to survive than those with 1-3 parents or children.

# AGE FEATURE

# In[12]:


#sort ages into logical categories
train['Age']=train['Age'].fillna(-0.5)
test['Age']=test['Age'].fillna(-0.5)
bins=[-1,0,5,12,18,24,35,60,np.inf]
labels=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup']=pd.cut(train['Age'],bins, labels=labels)
test['AgeGroup']=pd.cut(test['Age'],bins, labels=labels)

#draw a barplot of age vs survival
sns.barplot(x='AgeGroup', y='Survived',data=train)
plt.show()


# Babies are more likely to survive than any other group

# CABIN FEATURE

# In[14]:


train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=train)
plt.show()


# People with a recorded Cabin number are in fact more likely to survive (66.6% vs 29.9%)

# # 5. Cleaning Data

# Time to clean our data to account for missing values and unnecessary information
Looking at the test data
# In[15]:


test.describe(include='all')


# * We have total 418 passengers
# * 1 value from the fare feature is missing
# * Around 20.5% of the age feature is missing, we will need to fill that in

# # CABIN FEATURE

# In[16]:


#We'll start off by dropping the cabin feature since not a lot more useful information
train=train.drop(['Cabin'], axis=1)
test=test.drop(['Cabin'], axis=1)


# # TICKET FEATURE

# In[17]:


#We can also drop the ticket feature since its unlikely to yeild any useful information
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'], axis=1)


# # EMBARKED FEATURE

# In[18]:


#now we need to fill in the missing values in the embarked feature
print('Number of people embarking in Southampton (S): ')
southampton = train[train['Embarked']=='S'].shape[0]
print(southampton)

print('Number of people embarking in Cherboug (C): ')
cherboug = train[train['Embarked']=='C'].shape[0]
print(cherboug)

print('Number of people embarking in Queenstown (Q): ')
queenstown = train[train['Embarked']=='Q'].shape[0]
print(queenstown)


# Its clear that most of the people embarked from Southampton(S). Let's go ahead and fill in the missing values with S

# In[19]:


#replacing missing values in the enbarked feature with S
train=train.fillna({'Embarked':'S'})


# # AGE FEATURE

# Next we will fill in the missing values in age feature. Since a higher percentage of values are missing, it would be illogical to fill all of them the same value. (as we did with embarked). Instead, let's try to find a way to predict the missing ages.

# In[20]:


#create a combined group of both datasets
combine=[train,test]

#extract a title for each name in the train and test datasets
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

    pd.crosstab(train['Title'],train['Sex'])


# In[21]:


#replace various titles with more common names

for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Capt','Col',
                                              'Don','Dr','Major','Rev','Jonkeer','Dona'],'Rare')
    
    dataset['Title']=dataset['Title'].replace(['Countless','Lady','Sir'],'Royal')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')
    
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()


# In[22]:


#map each of the title groups to a numerical value
title_mapping={'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
    
    train.head()


# Next, we will try to predict the missing Age values from the most common age from their title

# In[23]:


# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#I tried to get this code to work with using .map(), but couldn't.
#I've put down a less elegant, temporary solution for now.
#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]


# Now that we've filled in the missing values at least somewhat accurately (I will work on a better way for predicting missing age values), it's time to map each age group to a numerical value.

# In[24]:


#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# # Name Feature

# We can drop the name feature now that we've extracted the titles.

# In[25]:


#drop the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# # Sex Feature

# In[26]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# # Embarked Feature

# In[27]:


#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# # Fare Feature

# It's time separate the fare values into some logical groups as well as filling in the single missing value in the test dataset.

# In[28]:


#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[29]:


#check train data
train.head()


# In[30]:


#check test data
test.head()


# # 6) Choosing the Best Model

# # Splitting the Training Data

# We will use part of our training data (22% in this case) to test the accuracy of our different models.

# In[31]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# # Testing Different Models

# I will be testing the following models with my training data :
# 
# * Gaussian Naive Bayes
# * Logistic Regression
# * Support Vector Machines
# * Perceptron
# * Decision Tree Classifier
# * Random Forest Classifier
# * KNN or k-Nearest Neighbors
# * Stochastic Gradient Descent
# * Gradient Boosting Classifier
# 
# For each model, we set the model, fit it with 80% of our training data, predict for 20% of the training data and check the accuracy.

# In[32]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[33]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[34]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[35]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[36]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[37]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[38]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[39]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[40]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[41]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# Let's compare the accuracies of each model!

# In[42]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# I decided to use the Gradient Boosting Classifier model for the testing data.

# # 7) Creating Submission File

# It's time to create a submission.csv file to upload to the Kaggle competition!

# In[44]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanicpredsubmission.csv', index=False) 


# In[ ]:




