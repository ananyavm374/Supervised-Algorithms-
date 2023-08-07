#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd 
df=pd.read_csv("//home/ubuntu/Downloads/heart.csv")
df.head()


# In[14]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[16]:


from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()
m.fit(x_train)
x_train=m.fit_transform(x_train)
x_test=m.fit_transform(x_test)


# In[17]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(model.predict([[53,1,0,140,212,0,1,155,0,1.0,2,2,2]]))


# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# score=accuracy_score(y_test,y_pred)
# print(score)
# print(classification_report(y_test,y_pred))

# In[ ]:




