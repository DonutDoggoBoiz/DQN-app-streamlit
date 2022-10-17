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
  symbol_list = df['Symbol'].to_list()
  company_list = df['Company'].to_list()
  market_list = df['Market'].to_list()
  industry_list = df['Industry'].to_list()
  sector_list = df['Sector'].to_list()
  stock_full_dict = {'symbol':symbol_list,
            'company_name':company_list,
            'market':market_list,
            'industry':industry_list,
            'sector':sector_list}
  stock_db.put(stock_full_dict)
