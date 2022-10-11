# Contents of ~/my_app/main_page.py
import streamlit as st
import numpy as np
import pandas as pd

### ------ USER DATABASE ------ ###
user_list = []
user1 = {'username':'admin', 'password':'admin'}
user2 = {'username':'admin2','password':'admin2'}
user_list.append(user1)
user_list.append(user2)
user_df = pd.DataFrame(user_list)

### ------ MODEL DATABASE ------ ###
model_list = []
model1 = {'username':'admin', 'model_name':'bbl_01', 'stock_quote':'bbl'.upper()}
model2 = {'username':'admin', 'model_name':'ptt_04', 'stock_quote':'ptt'.upper()}
model_list.append(model1)
model_list.append(model2)
model_df = pd.DataFrame(model_list)


# --- USER AUTHENTICATION --
st.markdown("# Welcome to our Home page 🎉")
st.markdown("💸💸💸")
#st.sidebar.markdown("# Home page 🎉")
