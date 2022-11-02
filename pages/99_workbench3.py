### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time
#streamlit-aggrid
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
### ---------------------- ###

if 'del_mod_button_status' not in st.session_state:
  st.session_state['del_mod_button_status'] = False
  
### ---------------------- ###


datamodel_dict = {'model_name': ['bbl_01','bbl_02','ppt_05','scg_111','mint_01'],
             'gamma': [0.90,0.80,0.85,0.75,0.95],
             'learning_rate': [0.001,0.002,0.005,0.04,0.099],
             'initial_balance': [1000000,1200000,1980000,2550000,3390000],
             'trading_size': [0.10,0.25,0.15,0.30,0.50]
            }
datamodel_df = pd.DataFrame(datamodel_dict)
st.dataframe(datamodel_df)

#-------
gb = GridOptionsBuilder.from_dataframe(datamodel_df)
#gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
#gb.configure_side_bar() #Add a sidebar
gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[0])
gridoptions = gb.build()

grid_response = AgGrid(datamodel_df,
                       fit_columns_on_grid_load=True,
                       gridOptions=gridoptions)

grp_data = grid_response['data']
selected_row = grid_response['selected_rows'] 

placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()

with placeholder2.container():
  ph2col1, ph2col2, _ = st.columns([1,1,4])
  with ph2col1:
    edit_mod_button = st.button('Edit')
  with ph2col2:
    del_mod_button = st.button('Delete')
    
if del_mod_button or st.session_state['del_mod_button_status']:
  st.session_state['del_mod_button_status'] = True
  with placeholder3.container():
    with st.form('del_make_sure'):
      st.write('Are you sure?')
      make_sure_radio = st.radio('Please confirm your choice:', 
                                 options=('No', 'Yes') )
      confirm_button = st.form_submit_button('Confirm')
      if confirm_button:
        if make_sure_radio == 'Yes':
          st.session_state['del_mod_button_status'] = False
          st.error('Model {} has been successfully deleted'.format(selected_row[0]['model_name']))
          time.sleep(3)
          st.experimental_rerun()
        elif make_sure_radio == 'No':
          st.session_state['del_mod_button_status'] = False
          placeholder3.empty()
  
#if selected_data == False:
#  placeholder.empty()
#else:
try:
  placeholder.empty()
  with placeholder.container():
    with st.expander('More model information:'):
        st.write('Name : {}'.format(selected_row[0]['model_name']))
        st.write('Gamma : {:.2f}'.format(selected_row[0]['gamma']))
        st.write('Learning Rate : {:.3f}'.format(selected_row[0]['learning_rate']))
        st.write('Initial Balance : {:,} THB'.format(selected_row[0]['initial_balance']))
        st.write('Trading Size : {:.2f}%'.format(selected_row[0]['trading_size']*100))
except:
  with placeholder.container():
    st.success('Loading...')
        
