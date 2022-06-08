#!/usr/bin/env python
# coding: utf-8

# # import libarary

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pickle


# ### read df_train

# In[13]:


df = pd.read_csv('E:/cs project/elg7186-project-group_project_-5-main/elg7186-project-group_project_-5-main/data/row/df_train.csv' , encoding='cp1252')


# In[14]:


df


# ### def for preprossening

# In[15]:


def preprossening_all(df , im_feature):
    df_new = df.drop(['Bwd PSH Flags' ,'Bwd URG Flags', 'Fwd Avg Bytes/Bulk' , 'Fwd Avg Packets/Bulk',
                        'Bwd Avg Bytes/Bulk' , 'Bwd Avg Packets/Bulk',
                          'Flow ID', 'Source IP', 'Source Port',
                           'Destination IP', 'Destination Port', 
                          'Protocol', 'Timestamp','Fwd Avg Bulk Rate', 'Bwd Avg Bulk Rate'], axis=1)
    
    df_clean = df_new.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    
    df_im = df_clean[im_feature]
    df_im['Label'] = df_clean.Label
    
    return df_im 
    


# ### importance feature

# In[16]:


im_feature = ['PSH Flag Count','Init_Win_bytes_forward','Bwd Packet Length Std','Avg Bwd Segment Size',
                'Bwd Packet Length Max','Packet Length Variance','Bwd Packet Length Mean','Bwd Packet Length Min','Packet Length Std',
                'min_seg_size_forward','Average Packet Size','Flow IAT Max','Fwd IAT Std','Idle Min','Packet Length Mean',
                  'Fwd IAT Max','Idle Max','ACK Flag Count','Flow IAT Std','URG Flag Count']


# In[17]:


df_final = preprossening_all(df, im_feature)


# In[18]:


df_final


# ### split df_train to train & val

# In[19]:


X = df_final.iloc[:,:-1]
y = df_final.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 


# ### train catboostclassifier

# In[20]:


xgboost_model = XGBClassifier(random_state=42)
xgboost_model.fit(X_train, y_train)
y_pred_xg = xgboost_model.predict(X_test)
f_xg = f1_score(y_test, y_pred_xg, average='macro')
acc_xg = accuracy_score(y_test, y_pred_xg)
acc_xg


# In[21]:


print(classification_report(y_test, y_pred_xg))
confusion_matrix = confusion_matrix(y_test, y_pred_xg)
ConfusionMatrixDisplay(confusion_matrix, display_labels = xgboost_model.classes_).plot()


# In[23]:


pickle.dump(xgboost_model, open('E:/cs project/elg7186-project-group_project_-5-main/elg7186-project-group_project_-5-main/models/xgboost_model.pkl', 'wb'))


# In[ ]:




