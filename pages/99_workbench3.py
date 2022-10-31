### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time
#streamlit-aggrid
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
### ---------------------- ###

data_dict = {'model_name': ['bbl_01','bbl_02','ppt_05','scg_111','mint_01'],
             'gamma': [0.90,0.80,0.85,0.75,0.95],
             'learning_rate': [0.001,0.002,0.005,0.04,0.099],
             'initial_balance': [1000000,1200000,1980000,2550000,3390000],
             'trading_size': [0.10,0.25,0.15,0.30,0.50]
            }
data_df = pd.DataFrame(data_dict)
st.dataframe(data_df)

#-------
gb = GridOptionsBuilder.from_dataframe(data_df)
#gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
#gb.configure_side_bar() #Add a sidebar
gb.configure_selection('single', use_checkbox=True)
gridoptions = gb.build()

grid_response = AgGrid(data_df,
                       gridOptions=gridoptions)

data = grid_response['data']
selected_data = grid_response['selected_rows'] 
#df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
st.write(selected_data)
with st.expander('data'):
  st.write(data)
