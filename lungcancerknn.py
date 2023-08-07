#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
df=pd.read_csv("//home/ubuntu/Downloads/lungcancer.csv")
df.head()


# In[3]:


x=df.iloc[:,2:-1].values


# In[7]:


y=df.iloc[:,-1].values


# In[12]:


from collections import Counter
print("the number of classes",Counter(y))


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[13]:


from sklearn import preprocessing
m=preprocessing.MinMaxScaler()
x_train=m.fit_transform(x_train)
x_test=m.fit_transform(x_test)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train,y_train)


# In[15]:


y_pred=classifier.predict(x_test)
print(y_pred)


# In[17]:


from sklearn.metrics import confusion_matrix,accuracy_score
res=accuracy_score(y_test,y_pred)
print(res)


# In[ ]:




