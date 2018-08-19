
# coding: utf-8
Problem : https://www.hackerearth.com/challenge/hiring/affine-analytics-ml-challenge/machine-learning/targeting-strategy-to-market-the-right-properties/

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel



# In[31]:


property_df=pd.read_csv('Properties.csv')


# In[19]:


property_df


# In[9]:


property_df.sale_date__c.unique()


# In[18]:


account_property_df=pd.read_csv('Accounts_properties.csv')
len(account_property_df)


# In[26]:


account_property_df.groupby(['id_props']).count()


# In[28]:


account_property_df[account_property_df.id_props.eq('a0I2A00000XQGKlUAP')]


# In[23]:


account_property_df.merge(property_df,how='right',on='id_props')


# In[38]:


property_new_df=property_df.drop(['sale_date__c'],axis=1)
property_ar =property_new_df.values


# In[39]:


property_ar


# In[61]:


property_type_dict = dict()
region_dict=dict()
pt_cnt=0
rg_cnt=0
for idx, row in property_df.iterrows():
    prop_type = row['property_type_1']
    region = row['region__c']
    pt_value = property_type_dict.get(prop_type, -1)
    rg_value = region_dict.get(region, -1)
    
    if pt_value!=-1:
        row['property_type_1']=pt_value
        property_df.iloc[idx, property_df.columns.get_loc('property_type_1')] = pt_value
    else:
        row['property_type_1']=pt_cnt
        property_df.iloc[idx, property_df.columns.get_loc('property_type_1')] = pt_cnt
        pt_cnt+=1
    if rg_value!=-1:
        row['region__c']=rg_value
        property_df.iloc[idx, property_df.columns.get_loc('region__c')] = rg_value
    else:
        row['region__c']=rg_cnt
        property_df.iloc[idx, property_df.columns.get_loc('region__c')] = rg_cnt
        rg_cnt+=1
        

property_new_df = property_df.apply(pd.to_numeric, errors='coerce')

property_new_df = property_new_df.fillna(0)

property_new_df


# In[66]:


property_ar =property_new_df.values
cosine_similarity_matrix=linear_kernel(property_ar,property_ar)
cosine_similarity_matrix.shape


# In[81]:


account_dict = dict()
unique_account_ids = account_property_df.id_accs.unique()
for account in unique_account_ids:
    tdf = account_property_df[account_property_df.id_accs.eq(account)]
    similar_properties=[]
    for prop in tdf.id_props.unique():
        #print (prop)
        try:
            prop_idx = property_df.loc[property_df.id_props == prop].index[0]
            #print (prop_idx)
            similar_indices = cosine_similarity_matrix[prop_idx].argsort()[:-5:-1]
            #print (similar_indices)
            similar_items = [(cosine_similarity_matrix[prop_idx][i], property_df['id_props'][i]) for i in similar_indices]
            similar_properties.append(similar_items)
        except IndexError:
            {}
            #print ('Property ', prop, 'does not exists')
    print (account)        
    #print (similar_properties)    
    account_dict[account] = similar_properties


# In[86]:


account_dict


# In[83]:


test_data_df = pd.read_csv('Test_Data.csv')
test_data_df.head()


# In[85]:


len(test_data_df)


# In[84]:


for idx, row in test_data_df.iterrows():
    val = account_dict.get(row['id_accs'],-1)
    if val==-1:
        print row['id_accs']

