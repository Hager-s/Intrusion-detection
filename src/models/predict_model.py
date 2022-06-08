#!/usr/bin/env python
# coding: utf-8

# # import libarary

# In[28]:


import pickle 
import numpy as np


# In[18]:


def preprossening_all(df , im_feature):
    df_new = df.drop(['Bwd PSH Flags' ,'Bwd URG Flags', 'Fwd Avg Bytes/Bulk' , 'Fwd Avg Packets/Bulk',
                        'Bwd Avg Bytes/Bulk' , 'Bwd Avg Packets/Bulk',
                          'Flow ID', 'Source IP', 'Source Port',
                           'Destination IP', 'Destination Port', 
                          'Protocol', 'Timestamp','Fwd Avg Bulk Rate', 'Bwd Avg Bulk Rate'], axis=1)
    
    df_clean = df_new.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    
    df_im = df_clean[im_feature]
    # df_im['Label'] = df_clean.Label
    
    return df_im 
    


# In[26]:


def predict_data(df):
    
    xg_boost = pickle.load(open("E:/cs project/elg7186-project-group_project_-5-main/elg7186-project-group_project_-5-main/models/xgboost_model.pkl", 'rb'))
    im_feature = ['PSH Flag Count','Init_Win_bytes_forward','Bwd Packet Length Std',
                    'Avg Bwd Segment Size',
                    'Bwd Packet Length Max',
                    'Packet Length Variance',
                    'Bwd Packet Length Mean',
                      'Bwd Packet Length Min',
                    'Packet Length Std',
                    'min_seg_size_forward',
                    'Average Packet Size',
                    'Flow IAT Max',
                    'Fwd IAT Std',
                    'Idle Min',
                    'Packet Length Mean',
                    'Fwd IAT Max',
                    'Idle Max',
                    'ACK Flag Count',
                    'Flow IAT Std',
                    'URG Flag Count']
    
    data = preprossening_all(df , im_feature)

    for i in data.columns:
        data[i][0]=int(float(data[i][0]))

    data=data.astype("int64")
    y_pred = xg_boost.predict(data)
    
    y_pred_proba = xg_boost.predict_proba(data)
    pred_proba=list(y_pred_proba)

    y_pred_proba=[]
    for i in pred_proba:
        y_pred_proba.append(np.max(i))

    
    return y_pred , y_pred_proba


# In[ ]:




