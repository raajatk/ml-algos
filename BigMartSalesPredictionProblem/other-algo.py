#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.columns


# In[3]:


train.isnull().sum()


# In[4]:


test.isnull().sum()


# In[5]:


train.apply(lambda x: len(x.unique()))
test.apply(lambda x: len(x.unique()))


# In[6]:


import copy
train_cat = train.select_dtypes(include=['object']).copy()
test_cat = test.select_dtypes(include=['object']).copy()


# In[7]:


train_cat.apply(lambda x: len(x.unique()))
test_cat.apply(lambda x:len(x.unique()))


# In[8]:


for x in train_cat:
    if(x!='Item_Identifier' and x!='Outlet_Identifier'):
        print("\nFrequency for category ",x)
        print(train_cat[x].value_counts())
    


# In[9]:


train['Item_Weight']=train['Item_Weight'].fillna(train['Item_Weight'].mean())
test['Item_Weight']=test['Item_Weight'].fillna(test['Item_Weight'].mean())


# In[10]:


train['Item_Weight'].describe()


# In[11]:


test['Item_Weight'].describe()


# In[12]:


train['Outlet_Size']=train['Outlet_Size'].fillna(train['Outlet_Size'].value_counts().index[0])
test['Outlet_Size']=test['Outlet_Size'].fillna(test['Outlet_Size'].value_counts().index[0])


# In[13]:


train.isnull().sum()


# In[14]:


test.isnull().sum()


# In[15]:


train.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')


# In[16]:


visibility_avg = train.pivot_table(values='Item_Visibility', index='Item_Identifier')

print(visibility_avg.xs('DRA12'))
#Impute 0 values with mean visibility of that product:
miss_bool = (train['Item_Visibility'] == 0)

print('Number of 0 values initially: %d'%sum(miss_bool))
train.loc[miss_bool,'Item_Visibility'] = train.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.xs(x))
print('Number of 0 values after modification: %d'%sum(train['Item_Visibility'] == 0))


# In[17]:


visibility_avg_test = test.pivot_table(values='Item_Visibility', index='Item_Identifier')

print(visibility_avg.xs('DRA12'))
#Impute 0 values with mean visibility of that product:
miss_bool = (test['Item_Visibility'] == 0)

print('Number of 0 values initially: %d'%sum(miss_bool))
test.loc[miss_bool,'Item_Visibility'] = test.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg_test.xs(x))
print('Number of 0 values after modification: %d'%sum(test['Item_Visibility'] == 0))


# In[18]:


train


# In[19]:


test


# In[20]:


data = train


# In[21]:


data


# In[22]:


data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[23]:


test['Item_Identifier'].value_counts()
test['Item_Type_Combined'] = test['Item_Identifier'].apply(lambda x: x[0:2])
test['Item_Type_Combined'] = test['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
test['Item_Type_Combined'].value_counts()


# In[24]:


test


# In[25]:


data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[26]:


test['Outlet_Years'] = 2013 - test['Outlet_Establishment_Year']
test['Outlet_Years'].describe()


# In[27]:


print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())


# In[28]:


print('Original Categories:')
print(test['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(test['Item_Fat_Content'].value_counts())


# In[29]:


data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[30]:


data['Item_Fat_Content']


# In[31]:


test.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
test['Item_Fat_Content'].value_counts()


# In[32]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[34]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
test['Outlet'] = le.fit_transform(test['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    test[i] = le.fit_transform(test[i])


# In[35]:


data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])


# In[36]:


test = pd.get_dummies(test, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])


# In[37]:


data.dtypes


# In[38]:


data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[39]:


test[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[41]:


#Drop the columns which have been converted to different types:
# data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

# #Divide into test and train:
# train = data.loc[data['source']=="train"]
# test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
# test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
# train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
# data.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)
# test.to_csv("test_modified.csv",index=False)


# In[ ]:




