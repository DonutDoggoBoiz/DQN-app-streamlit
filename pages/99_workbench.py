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
