#!/usr/bin/env python
# coding: utf-8

# # IQVIA - SEGMENTATION

# ### IMPORT LIBRARIES & DATA

# In[40]:


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


df0 = pd.read_excel(r'C://Users/Namrata/Desktop/Q2_Dummy data - EXERCISE.xlsx')
df0.shape


# In[3]:


df0.head()


# In[4]:


df0.describe()


# In[14]:


df0.columns


# In[15]:


df0['Patient country'].value_counts()


# ## RFM Analysis

# In[24]:


import datetime
recent_date = max(df0.claim_date) + datetime.timedelta(days=1) 

    
# Aggregate data by each patient
patients = df0.groupby(['patient_id']).agg({
    'claim_date': lambda x: (recent_date - x.max()).days,
    'claim_id': 'count',
    'Total money spent': 'sum'})
# Rename columns
patients.rename(columns = {'claim_date': 'Recency',
                            'claim_id': 'Frequency',
                            'Total money spent': 'MonetaryValue'}, inplace=True)


# In[25]:


patients


# In[34]:


fig, ax = plt.subplots(1, 3, figsize=(15,3))
sns.distplot(patients['Recency'], ax=ax[0])
sns.distplot(patients['Frequency'], ax=ax[1])
sns.distplot(patients['MonetaryValue'], ax=ax[2])
plt.tight_layout()
plt.show()


# In[35]:


from scipy import stats
def patients_skewness(x):
    fig, ax = plt.subplots(2, 2, figsize=(5,5))
    sns.distplot(patients[x], ax=ax[0,0])
    sns.distplot(np.log(patients[x]), ax=ax[0,1])
    sns.distplot(np.sqrt(patients[x]), ax=ax[1,0])
    sns.distplot(stats.boxcox(patients[x])[0], ax=ax[1,1])
    plt.tight_layout()
    plt.show()
    
    print(patients[x].skew().round(2))
    print(np.log(patients[x]).skew().round(2))
    print(np.sqrt(patients[x]).skew().round(2))
    print(pd.Series(stats.boxcox(patients[x])[0]).skew().round(2))


# In[36]:


patients_skewness('Recency')


# In[37]:


patients_skewness('Frequency')


# In[38]:


patients_skewness('MonetaryValue')


# ### Variable transformation for skewness

# In[39]:


# Set the Numbers

patients_tf1 = pd.DataFrame()
patients_tf1["Recency"] = stats.boxcox(patients['Recency'])[0]
patients_tf1["Frequency"] = stats.boxcox(patients['Frequency'])[0]
patients_tf1["MonetaryValue"] = stats.boxcox(patients['MonetaryValue'])[0]
patients_tf1.tail()


# ### Standardization

# In[41]:


scaler = StandardScaler()
scaler.fit(patients_tf1)
patients_normalized = scaler.transform(patients_tf1)
print(patients_normalized.mean(axis = 0).round(2))
print(patients_normalized.std(axis = 0).round(2))


# In[42]:


pd.DataFrame(patients_normalized).head()


# ## Clustering

# ### Choose k- number

# In[43]:


from sklearn.cluster import KMeans

sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=666)
    kmeans.fit(patients_normalized)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[44]:


model = KMeans(n_clusters=3, random_state=666)
model.fit(patients_normalized)
model.labels_.shape


# In[45]:


patients.shape


# ### Analysis

# In[46]:


patients['cluster'] = model.labels_
patients.head()


# In[48]:


patients.groupby('cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(1)


# ### Snake plot - to check how clusters differ from each other

# In[50]:


df_normalized = pd.DataFrame(patients_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = patients.index
df_normalized['cluster'] = model.labels_
df_normalized.head()


# In[51]:


#melt
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()


# In[52]:


sns.lineplot('Attribute', 'Value', hue='cluster', data=df_nor_melt)


# In[53]:


patients.groupby('cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(1)


# In[54]:


cluster_avg = patients.groupby('cluster').mean()
population_avg = patients.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp


# ### Inference
# 
# #### Cluster 0-  less frequent, less money spent but recently claimed. New patients
# #### Cluster 1-  more frequent, more money spent and recently claimed. High value patients 
# #### Cluster 2-  less frequent, less money spent and claimed long ago. Low value patients
# 
# 
# 

# In[ ]:




