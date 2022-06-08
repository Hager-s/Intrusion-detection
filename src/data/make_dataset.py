#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kafka import KafkaConsumer
import pandas as pd
import sys
#sys.path.append('..')
sys.path.insert(0, 'E:\cs project\elg7186-project-group_project_-5-main\elg7186-project-group_project_-5-main')
from src.models.predict_model import predict_data
from src.models.predict_model import preprossening_all
from elasticsearch import Elasticsearch, helpers
from kafka import KafkaConsumer
from kafka import KafkaProducer
import pickle 
import numpy as np
import datetime

# In[ ]:

column = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp',
            'Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets',
            'Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean',
            'Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean',
           'Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max',
           'Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min',
           'Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags',
           'Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s',
           'Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance',
           'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count',
           'ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length.1',
           'Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk',
           'Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes',
           'Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean',
           'Active Std','Active Max', 'Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label']

row_cols = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol']


# In[ ]:


def create_dataframe(consumer, column, row_cols):
   
    all_record = []
    all_df, row = [], []
    df = pd.DataFrame(columns= column)
    df_row = pd.DataFrame(columns= row_cols)
    for i,d in enumerate(consumer):
        txt = d.value.decode()
        new_record = txt.split(',')
        row_record = new_record[:6]
        all_df.append(new_record)
        df = pd.DataFrame(all_df, columns=column)
        y_pred, y_score = predict_data(df)
        print(f'\n\{txt}n\n{df}\n\n{y_pred}\n\n\n')
        row.append(row_record)
        df_row_med = pd.DataFrame(row, columns= row_cols)
        df_row_med['pred'] = y_pred[0]
        df_row_med['score'] = y_score[0]
        df_row_med['timestamp'] = str(datetime.datetime.now().replace(microsecond=0))

        if y_pred == 1:
            helpers.bulk(elastic_client, df_row_med.transpose().to_dict().values(), index="intrusion-predictions")
        if len(all_df) > 10000:
            break

        all_df = []


    # return df_row


# In[ ]:
elastic_client = Elasticsearch(
    "https://localhost:9200",
    http_auth=("elastic", "611997nada"),ca_certs="E:\cs project\elg7186-project-group_project_-5-main\elasticsearch-8.1.2\config\certs\http_ca.crt"
    ,  verify_certs=False
)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         json.dumps(x).encode('utf-8'))




def main():
    
    consumer = KafkaConsumer(
    'Intrusion_detection',
    bootstrap_servers = ['localhost:9092'],
    auto_offset_reset = 'earliest',
    enable_auto_commit = False)
    create_dataframe(consumer, column, row_cols)

if __name__ == '__main__':
    
    main()
    #final_out = main()



# In[ ]:




