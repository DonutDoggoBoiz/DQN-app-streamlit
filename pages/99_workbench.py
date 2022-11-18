### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta

csv_path = 'models/mnist_test.csv'
df_csv = pd.read_csv(csv_path)

show_csv = st.button('Show csv')
if show_csv:
  st.write(csv_path)
  st.dataframe(df_csv)
  

# streamlit_app.py

import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)
path_uri = 'gs://streamlitapphost.appspot.com/gcs_mnist_test.csv'
bucket_name = "streamlitapphost.appspot.com"
bucket = client.bucket(bucket_name)

file_path = 'gcs_mnist_test.csv'

gcs_switch = st.checkbox('GCS Switch')
if gcs_switch:
  local_path = 'models/gcs_mnist_test.csv'
  local_path2 = 'models/gcs_mnist_test2.csv'
  content = bucket.blob(file_path).download_to_filename(local_path)
  content2 = bucket.blob(file_path).download_as_bytes()

show_gcs_file = st.button('Show GCS file')
show_local_file = st.button('Show local file')

if show_gcs_file:
  try:
    #gcs_df = pd.read_csv(content2)
    #st.dataframe(gcs_df)
    st.write('content2 is :')
    st.write(type(content2))
    with st.expander('show string'):
      st.write(content2)
  except:
    st.error('ERROR GCS')
    
if show_local_file:
  try:
    local_df = pd.read_csv(local_path)
    st.dataframe(local_df)
  except:
    st.error('ERROR LOCAL')
