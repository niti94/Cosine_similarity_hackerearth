
# coding: utf-8
Problem : https://www.hackerearth.com/challenge/hiring/affine-analytics-ml-challenge/machine-learning/targeting-strategy-to-market-the-right-properties/


# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from functools import reduce
from __future__ import division


# In[195]:


property_df=pd.read_csv('Properties.csv')


# In[158]:


property_df


# In[169]:


proplist = property_df['id_props'].unique().tolist()
print len(proplist)


# In[172]:


account_property_df=pd.read_csv('Accounts_properties.csv')
accproplist = account_property_df['id_props'].unique().tolist()
print (len(accproplist))
set_prop_list = set(accproplist)
unbought_properties = []
for prop in proplist:
    if prop in set_prop_list:
        unbought_properties.append(prop)
print len(unbought_properties)        


# In[26]:


account_property_df.groupby(['id_props']).count()


# In[28]:


account_property_df[account_property_df.id_props.eq('a0I2A00000XQGKlUAP')]


# In[23]:


account_property_df.merge(property_df,how='right',on='id_props')


# In[181]:


property_new_df=property_df.drop(['sale_date__c'],axis=1)
property_ar =property_new_df.values


# In[182]:


property_ar


# In[183]:


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


# In[187]:


property_ar =property_new_df.values
prop_euclidean_distances_matrix = euclidean_distances(property_ar,property_ar)
print prop_euclidean_distances_matrix.shape


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


# In[82]:


account_df=pd.read_csv('Accounts.csv')
account_df.head()


# In[85]:


account_dict=dict()
ac_cnt=0
for idx, row in account_df.iterrows():
    account_type = row['investor_type']
   # region = row['region__c']
    account_value = account_dict.get(account_type, -1)
    #account_value = region_dict.get(region, -1)
    
    if account_value!=-1:
        #row['investor_type']=account_value
        account_df.iloc[idx, account_df.columns.get_loc('investor_type')] = account_value
    else:
        #row['investor_type']=ac_cnt
        account_df.iloc[idx, account_df.columns.get_loc('investor_type')] = ac_cnt
        ac_cnt+=1


# In[134]:


account_df.head()


# In[238]:


train, test1 = train_test_split(account_df, test_size=0.15)
cv, test = train_test_split(test1, test_size=0.5)
print (len(account_df))
print (len(train))
print (len(cv))
print (len(test))
print (len(train) + len(cv) + len(test))


# In[136]:


account_new_df=account_df.drop(['id_accs'],axis=1)
#account_new_df=account_new_df.drop(['activity_count'],axis=1)
account_new_df = account_new_df.applymap(lambda x: 0 if x == False else x)
account_new_df = account_new_df.applymap(lambda x: 1 if x == True else x)
#print (account_new_df.head())

account_ar =account_new_df.values
euclidean_distances_matrix=euclidean_distances(account_ar,account_ar)
euclidean_distances_matrix.shape
print account_ar


# In[137]:


print (euclidean_distances_matrix[12].argsort())


# In[219]:


test_data_df = pd.read_csv('Test_Data.csv')
test_data_df.head()


# In[132]:


with open('answer.csv', 'w') as the_file:
    the_file.write("id_accs,id_prop\n")
    for idx, row in test_data_df.iterrows():
        test_account = row['id_accs']
        #print (test_account)
        test_account_index = account_df.loc[account_df.id_accs == test_account].index[0]
        #print (test_account_index)
        test_account_similar_indices = euclidean_distances_matrix[test_account_index].argsort()[1:3000]
        #print (test_account_similar_indices)
        test_account_similar_accounts = [account_df['id_accs'][i] for i in test_account_similar_indices]
        #print (test_account_similar_accounts)
    #     if len(test_account_similar_accounts) == 0:
    #         print ('Account has no similar accounts', test_account)
    #     else:
    #         print (test_account)
        set_similar_accounts = set(test_account_similar_accounts)
        prop_list=[]
        for idy, aprow in account_property_df.iterrows():
            if len(prop_list) >= 30000:
                break
            if aprow['id_accs'] in set_similar_accounts:
                prop_list.append(aprow['id_props'])
        #print (prop_list)  
        for prop in prop_list:
            #print (test_account,prop)
            the_file.write("%s,%s\n" % (test_account,prop))


# In[138]:


cv.head()


# In[144]:


def recommendedProperties(account_id, num_similar_accs, max_properties):
    #index of account_id in accounts table
    account_index = account_df.loc[account_df.id_accs == account_id].index[0]
    account_similar_indices = euclidean_distances_matrix[account_index].argsort()[1:num_similar_accs]
    account_similar_accounts = [account_df['id_accs'][i] for i in account_similar_indices]
    set_similar_accounts = set(account_similar_accounts)
    prop_list=[]
    for idx, row in account_property_df.iterrows():
        if len(prop_list) >= max_properties:
            break
        if row['id_accs'] in set_similar_accounts:
            prop_list.append(row['id_props'])
    return prop_list


# In[188]:


def similarProperties(property_id, num_similar_props):
    property_index = property_df.loc[property_df.id_props == property_id].index[0]
    property_similar_indices = prop_euclidean_distances_matrix[property_index].argsort()[1:num_similar_props]
    property_similar_properties = [property_df['id_props'][i] for i in property_similar_indices]
    return property_similar_properties


# In[231]:


def recommendedPropertiesHybrid(account_id, num_similar_accs, max_properties, max_similar_properties):
    #index of account_id in accounts table
    account_index = account_df.loc[account_df.id_accs == account_id].index[0]
    account_similar_indices = euclidean_distances_matrix[account_index].argsort()[1:num_similar_accs]
    account_similar_accounts = [account_df['id_accs'][i] for i in account_similar_indices]
    set_similar_accounts = set(account_similar_accounts)
    prop_list=[]
    for idx, row in account_property_df.iterrows():
        if len(prop_list) >= max_properties:
            break
        if row['id_accs'] in set_similar_accounts:
            #prop_list.append(row['id_props'])
            try:
                prop_list.extend(similarProperties(row['id_props'], max_similar_properties))
            except IndexError:
                ()
    return prop_list


# In[201]:


def F1Score(bought_properties, recommended_properties, _lambda):
    set_recommended_properties = set(recommended_properties)
    precision = 0
    recall = 0
    converted_properties = 0
    for prop in bought_properties:
        if prop in set_recommended_properties:
            converted_properties+=1
    precision = converted_properties/(len(recommended_properties) + _lambda)
    recall = converted_properties/(len(bought_properties) + _lambda)
    print (precision, converted_properties, recall, len(recommended_properties), len(bought_properties))
    if precision == 0 and recall ==0:
        f1Score = 0
    else:
        f1Score = (2 * precision * recall)/(precision + recall)
    return f1Score


# In[255]:


#For each account in cross validation set
f1ScoreList = []
for idx, row in cv.iterrows():
    #get account id
    test_account = row['id_accs']
    #print (test_account)
    #get properties bought by that account
    test_account_properties = account_property_df[account_property_df.id_accs.eq(test_account)]['id_props'].tolist()
    #get recommended properties
    test_account_recommended_properties = recommendedPropertiesHybrid(test_account, 2500, 1000, 2000)
    #print(test_account,len(test_account_properties), len(test_account_recommended_properties))
    f1Score = F1Score(test_account_properties, test_account_recommended_properties, 0.00001)
    f1ScoreList.append(f1Score)

f1ScoreMean = reduce((lambda x, y: x + y), f1ScoreList)
f1ScoreMean /= len(f1ScoreList)
print 'F1Score of the algorithm is ' + str(f1ScoreMean)


# In[271]:


with open('answer.csv', 'w') as the_file:
    the_file.write("id_accs,id_prop\n")
    #For each account in cross validation set
    for idx, row in test_data_df.iterrows():
        #get account id
        test_account = row['id_accs']
        #print (test_account)
        #get properties bought by that account
        #test_account_properties = account_property_df[account_property_df.id_accs.eq(test_account)]['id_props'].tolist()
        #get recommended properties
        test_account_recommended_properties = recommendedPropertiesHybrid(test_account, 2500, 1000, 840)
        #print(test_account,len(test_account_properties), len(test_account_recommended_properties))
        for prop in test_account_recommended_properties:
            #print (test_account,prop)
            the_file.write("%s,%s\n" % (test_account,prop))


# ## 
