#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# * Importing Libraries

# In[4]:


import numpy as n
import pandas as p
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# * Reading Data

# In[15]:


d = p.read_csv("wine.csv")
print(d)


# In[16]:


d.head()


# * Removing the id column

# In[17]:


d = d.iloc[:,:-1]
d


# In[18]:


#find missing values in dataset
d.isnull().sum()


# * Distribution of class label

# In[19]:


sn.countplot(x="quality", data=d)
plt.xlabel("Wine Quality (0-10 scale)")
plt.show()


# * Correlation analysis

# In[20]:


corrMatt=d.corr()
mask=np.array(corrMatt)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots()
fig.set_size_inches(10,20)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
plt.show()


# * Splitting class label from other features

# In[21]:


X=d.iloc[:, :-1].values
y=d.iloc[:, -1].values

# adding extra column because of Multiple linear regression
X=np.append(arr=np.ones((X.shape[0],1)), values=X, axis=1)

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# * Scaling the dataset

# In[22]:


sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# * Linear regression

# In[23]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[24]:


#Prediction
pred=regressor.predict(X_test)


# In[25]:


prediction1=pd.DataFrame(pred)
prediction1.head()


# * Displaying results

# In[27]:


plt.scatter(y_test,prediction1, c='r')
plt.xlabel('Actual Quantity')
plt.ylabel('Predicted Quantity')
plt.title('Predicted Quantity Vs Actual Quantity')
plt.show()


# In[ ]:




