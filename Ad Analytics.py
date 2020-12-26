#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


adv = pd.read_csv("advertising.csv")
adv.head()


# In[6]:


adv.info()


# In[7]:


adv.describe()


# In[8]:


import seaborn as sns


# In[13]:


sns.set_style('whitegrid')


# In[14]:


sns.histplot(data=adv,x="Age")


# In[15]:


sns.jointplot(data=adv,x='Age',y='Area Income')


# In[19]:


sns.jointplot(data=adv,x='Age',y='Daily Time Spent on Site',kind='kde')


# In[16]:


adv.columns


# In[20]:


sns.jointplot(data=adv,x='Daily Time Spent on Site',y='Daily Internet Usage')


# In[21]:


sns.pairplot(data=adv)


# In[24]:


adv.head()


# In[26]:


adv.columns


# In[28]:


X=adv[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male']]
y=adv['Clicked on Ad']


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# In[32]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# In[33]:


logmodel.fit(X_train,y_train)


# In[36]:


predictions=logmodel.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

