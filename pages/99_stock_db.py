### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta

### --- DATABASE CONNECTION --- ###
deta = Deta(st.secrets["deta_key"])
stock_db = deta.Base("stock_db")

### --- INTERFACE --- ###
uploaded_file = st.file_uploader("Choose a file")

if st.button('Show Dataframe'):
  df = pd.read_csv(uploaded_file)
  st.write(df[:20])
  st.write('...')
  

if st.button('Add to Deta'):
  df = pd.read_csv(uploaded_file)
  df_dict = df.to_dict()
  stock_db.put(df_dict)
