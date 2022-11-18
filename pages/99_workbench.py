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

bucket_name = "streamlitapphost.appspot.com"
bucket = client.bucket(bucket_name)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    return content

bucket_name = "streamlit-bucket"
file_path = 'gcs_mnist_test.csv'

gcs_switch = st.checkbox('GCS Switch')
if gcs_switch:
  local_path = 'models/gcs_mnist_test.csv'
  content = bucket.blob(file_path).download_to_filename(local_path)
  show_gcs_file = st.button('Show GCS file')
  show_local_file = st.button('Show local file')
  if show_gcs_file:
    st.dataframe(content)
  if show_local_file:
    st.dataframe(local_path)

