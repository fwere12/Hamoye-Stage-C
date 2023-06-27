#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading in the data
data1 = pd.read_csv("https://query.data.world/s/wh6j7rxy2hvrn4ml75ci62apk5hgae.csv")
data1.head()


# In[3]:


#check distribution of target variable 
data1['QScore'].value_counts() 


# In[4]:


#Check for missing values
data1.isnull().sum()


# In[5]:


#We drop the missing values
data1 = data1.dropna()
data1.isnull().sum()


# In[6]:


data1['QScore'].value_counts() 


# An obvious change in our target variable after removing the missing values is that there are only three classes left and from the distribution of the 3 classes, we can see that there is an obvious imbalance between the classes. 
# There are methods that can be applied to handle this imbalance such as oversampling and undersampling.
# Oversampling involves increasing the number of instances in the class with fewer instances while undersampling involves reducing the data points in the class with more instances.
# For now, we will convert this to a binary classification problem by combining class '2A'and '1A'. 

# In[7]:


data1[ 'QScore' ] = data1[ 'QScore' ].replace([ '1A' ], '2A' )
data1.QScore.value_counts() 


# In[8]:


data1_2A = data1[data1.QScore== '2A' ]
data1_3A = data1[data1.QScore== '3A' ].sample( 350 )
data1 = data1_2A.append(data1_3A) 


# In[9]:


import sklearn.utils
data1 = sklearn.utils.shuffle(data1)
data1 = data1.reset_index(drop= True )
data1.shape
data1.QScore.value_counts() 


# In[10]:


#More preprocessing
data1 = data1.drop(columns = ['country_code', 'country', 'year'])
X = data1.drop(columns = 'QScore')
y = data1['QScore']


# In[11]:


#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 0 )
y_train.value_counts()


# There is still an imbalance in the class distribution. For this, we use SMOTE only on the training data to handle this. 

# In[12]:


#Encode categorical variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_train.record = encoder.fit_transform(X_train.record)
X_test.record = encoder.transform(X_test.record) 


# In[13]:


import imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state= 1 )
X_train_balanced, y_balanced = smote.fit_resample(X_train, y_train)


# In[14]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_train_df = scaler.fit_transform(X_train_balanced.drop(columns=[ 'record' ]))
normalised_train_df = pd.DataFrame(normalised_train_df, columns = X.drop(columns = ['record']).columns)
normalised_train_df[ 'record' ] = X_train_balanced[ 'record' ] 


# In[15]:


X_test = X_test.reset_index(drop = True )
normalised_test_df = scaler.transform(X_test.drop(columns = ['record']))
normalised_test_df = pd.DataFrame(normalised_test_df,
columns = X_test.drop(columns = ['record']).columns)
normalised_test_df[ 'record' ] = X_test['record']


# In[16]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(normalised_train_df, y_balanced)

#returns
LogisticRegression(C= 1.0 , class_weight= None, dual= False, fit_intercept= True,
                   intercept_scaling= 1 , l1_ratio= None, max_iter= 100 ,
                   multi_class= 'auto', n_jobs= None, penalty= 'l2',
                   random_state= None, solver= 'lbfgs', tol= 0.0001, verbose= 0,
                   warm_start= False)


# Measuring Classification Performance

# In[17]:


#Cross-validation and accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(log_reg, normalised_train_df, y_balanced, cv= 5 , scoring= 'f1_macro' )
scores


# In[18]:


#Confusion Matrix
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix
new_predictions = log_reg.predict(normalised_test_df)
cnf_mat = confusion_matrix(y_true=y_test, y_pred=new_predictions, labels=['2A', '3A'])
cnf_mat


# In[19]:


#Accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=new_predictions)
print( 'Accuracy: {}' .format(round(accuracy* 100 ), 2 ))


# In[20]:


#Precision
precision = precision_score(y_true=y_test, y_pred=new_predictions, pos_label= '2A' )
print( 'Precision: {}' .format(round(precision* 100 ), 2 ))


# In[21]:


#Recall
recall = recall_score(y_true=y_test, y_pred=new_predictions, pos_label= '2A' )
print( 'Recall: {}' .format(round(recall* 100 ), 2 ))


# In[22]:


#F1-Score
f1 = f1_score(y_true=y_test, y_pred=new_predictions, pos_label= '2A' )
print( 'F1: {}' .format(round(f1* 100 ), 2 ))


# In[23]:


#K-Fold Cross Validation
from sklearn.model_selection import KFold
kf = KFold(n_splits= 5 )
kf.split(normalised_train_df)
f1_scores = []

#run for every split
for train_index, test_index in kf.split(normalised_train_df):
    X_train, X_test = normalised_train_df.iloc[train_index], normalised_train_df.iloc[test_index]
    y_train, y_test = y_balanced[train_index], y_balanced[test_index]
    model = LogisticRegression().fit(X_train, y_train)
    #save the result to list
    f1_scores.append(f1_score(y_true=y_test, y_pred=model.predict(X_test), pos_label= '2A' )*100)


# In[24]:


#Stratified K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 5 , shuffle = True , random_state = 1 )
f1_scores = []

#run for every split
for train_index, test_index in skf.split(normalised_train_df, y_balanced):
    X_train, X_test = np.array(normalised_train_df)[train_index], np.array(normalised_train_df)[test_index]
    y_train, y_test = y_balanced[train_index], y_balanced[test_index]
    model = LogisticRegression().fit(X_train, y_train)
    #save the result to list
    f1_scores.append(f1_score(y_true=y_test, y_pred=model.predict(X_test), pos_label= '2A' )) 


# In[25]:


#Leave One Out Cross Validation (LOOCV)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(LogisticRegression(), normalised_train_df, y_balanced, cv=loo,
                         scoring= 'f1_macro' )
average_score = scores.mean() * 100 

