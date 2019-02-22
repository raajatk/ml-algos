#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataf = pd.read_csv('train_modified.csv')
test = pd.read_csv('test_modified.csv')


# In[3]:


dataf.describe()


# In[4]:


test.describe()


# In[5]:


dataf.columns


# In[6]:


test.columns


# In[23]:


x = dataf.drop(['Outlet_Identifier','Item_Identifier','Item_Outlet_Sales'],axis=1)
x_test = test.drop(['Outlet_Identifier','Item_Identifier','Outlet_Establishment_Year','Item_Type'],axis=1)


# In[24]:


y = dataf.drop(['Item_Identifier', 'Item_Weight', 'Item_Visibility', 'Item_MRP',
       'Outlet_Identifier', 'Outlet_Years',
       'Item_Fat_Content_0', 'Item_Fat_Content_1', 'Item_Fat_Content_2',
       'Outlet_Location_Type_0', 'Outlet_Location_Type_1',
       'Outlet_Location_Type_2', 'Outlet_Size_0', 'Outlet_Size_1',
       'Outlet_Size_2', 'Outlet_Type_0', 'Outlet_Type_1', 'Outlet_Type_2',
       'Outlet_Type_3', 'Item_Type_Combined_0', 'Item_Type_Combined_1',
       'Item_Type_Combined_2', 'Outlet_0', 'Outlet_1', 'Outlet_2', 'Outlet_3',
       'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7', 'Outlet_8', 'Outlet_9'],axis=1)


# In[25]:


y.describe()


# In[26]:


x.describe()


# In[27]:


print(x_test.columns)
print(x.columns)


# In[28]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)


# In[29]:


y_pred = model.predict(x_test)


# In[30]:


y_pred


# In[35]:


y_pred[1,0]


# In[41]:


print(test['Item_Identifier'][0])
print(test['Outlet_Identifier'][0])
print(y_pred[0,0])


# In[45]:


test['Item_Identifier'].describe()


# In[46]:


result = []
for i in range(5681):
    val = []
    val.append(test['Item_Identifier'][i])
    val.append(test['Outlet_Identifier'][i])
    val.append(y_pred[i,0])
    result.append(val)


# In[47]:


result


# In[49]:


result_f = pd.DataFrame(result)


# In[52]:


result_f.to_csv('result.csv',index=False)


# In[ ]:




