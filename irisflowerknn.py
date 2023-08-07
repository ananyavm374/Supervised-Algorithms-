#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 


# In[5]:


df=pd.read_csv("//home/ubuntu/Downloads/iris.csv")
df.head()


# In[31]:


x=df.iloc[:,0:-1].values


# In[30]:


y=df. iloc[:,-1].values


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[32]:


from sklearn.preprocessing import StandardScaler
sclr=StandardScaler()
sclr.fit(x_train)
x_train=sclr.transform(x_train)
x_test=sclr.transform(x_test)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train,y_train)


# In[34]:


y_pred=classifier.predict(x_test)


# In[40]:


print(classifier.predict([[5.1,3.5,1.4,0.2]]))


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
result=confusion_matrix(y_test,y_pred)
result1=accuracy_score(y_test,y_pred)
print(result)
print(result1)

