### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time
#streamlit-aggrid
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
### ---------------------- ###
if 'username' not in st.session_state:
  st.session_state['username'] = 'mike1994'
  
if 'del_mod_button_status' not in st.session_state:
  st.session_state['del_mod_button_status'] = False
  
### ---------------------- ###
deta = Deta(st.secrets["deta_key"])
user_db = deta.Base("user_db")
model_db = deta.Base("model_db")
model_frame = pd.DataFrame(model_db.fetch().items)
model_frame2 = pd.DataFrame(model_db.fetch({'username':st.session_state['username']}).items)
### ---------------------- ###


#model_for_grid = pd.DataFrame(model_db.fetch({'username':'mike1994'}).items)
shuffle_col = ['model_name','episode_trained','stock_quote','start_date','end_date','initial_balance','trading_size_pct','commission_fee_pct','gamma',]
frame_for_grid = model_frame2.loc[:,shuffle_col]

gb = GridOptionsBuilder.from_dataframe(frame_for_grid)
gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[0])
gridoptions = gb.build()
grid_response = AgGrid(frame_for_grid,
                       fit_columns_on_grid_load=False,
                       gridOptions=gridoptions)
grid_data = grid_response['data']
selected_row = grid_response['selected_rows'] 

### ---------------------- ###

placeholder_4 = st.empty()

with placeholder_4.container():
  ph2col1, ph2col2, _ = st.columns([1,1,6])
  with ph2col1:
    edit_mod_button = st.button('Edit')
  with ph2col2:
    del_mod_button = st.button('Delete')
  ### --- edit button --- ###
  if edit_mod_button: #or st.session_state['edit_mod_button_status']:
    #st.session_state['edit_mod_button_status'] = True
    with placeholder_4.container():
        edit_form_col1, _ = st.columns([2,1])
        with edit_form_col1:
          with st.form('edit parameter form'):
            st.write("##### Model parameters")
            edt_agent_name = st.text_input("Model name: ", "model_01")
            edt_agent_gamma = st.slider("Gamma: ", 0.00, 1.00, 0.90)
            edt_agent_epsilon = st.slider("Starting epsilon (random walk probability): ", 0.00, 1.00, 1.00)
            edt_agent_epsilon_dec = st.select_slider("Epsilon decline rate (random walk probability decline): ",
                                                 options=[0.001,0.002,0.005,0.010], value=0.001)
            edt_agent_epsilon_end = st.slider("Minimum epsilon: ", 0.01, 0.10, 0.01)
            edt_agent_lr = st.select_slider("Learning rate: ", options=[0.001, 0.002, 0.005, 0.010], value=0.001)
            st.write("##### Trading parameters")
            edt_initial_balance = st.number_input("Initial account balance (THB):", min_value=0, step=1000, value=1000000)
            edt_trading_size_pct = st.slider("Trading size as a percentage of initial account balance (%):", 0, 100, 10)
            edt_commission_fee_pct = st.number_input("Commission fee (%):", min_value=0.000, step=0.001, value=0.157, format='%1.3f')
            edit_param_button = st.form_submit_button("Edit")
            if edit_param_button:
              st.success('Edit parameters successful!')
              time.sleep(3)
              st.experimental_rerun()

  ### --- delete button --- ###
  if del_mod_button or st.session_state['del_mod_button_status']:
    st.session_state['del_mod_button_status'] = True
    with placeholder_4.container():
      with st.form('del_make_sure'):
        st.write('Are you sure?')
        make_sure_radio = st.radio('Please confirm your choice:', 
                                   options=('No', 'Yes') )
        confirm_button = st.form_submit_button('Confirm')
        if confirm_button:
          if make_sure_radio == 'Yes':
            st.session_state['del_mod_button_status'] = False
            selected_model_name = selected_row[0]['model_name']
            st.write(selected_model_name)
            key_to_del = model_frame2.loc[model_frame2['model_name']==selected_model_name,'key'].to_list()[0]
            #model_db.delete()
            st.error('Model {} has been successfully deleted with key: {}'.format(selected_model_name, key_to_del))
            time.sleep(3)
            st.experimental_rerun()
          elif make_sure_radio == 'No':
            st.session_state['del_mod_button_status'] = False
            st.experimental_rerun()
            
placeholder_end = st.empty()
with placeholder_end.container():
  show_model_frame = st.button('Show model frame 1')
  show_model_frame2 = st.button('Show model frame 2')
  write_selected_row = st.button('Write selected row')
  selected_row_df = st.button('Write selected_row as df')
if show_model_frame:
  st.dataframe(model_frame)
if show_model_frame2:
  st.dataframe(model_frame2)
if write_selected_row:
  detour_key_pd = model_frame2.loc[model_frame2['model_name']==selected_row[0]['model_name'],'key']
  detour_key_list = detour_key_pd.to_list()
  detour_key_object = detour_key_list[0]
  st.write(selected_row[0]['model_name'])
  st.write(' ')
  st.write('selected_row model_name: {}'.format(selected_row[0]['model_name']))
  st.write('selected_row detour key panda: {}, type: {}'.format(detour_key_pd, type(detour_key_pd)))
  st.write('detour key list: {}, type: {}'.format(detour_key_list, type(detour_key_list)))
  st.write('detour key object: {}, type: {}'.format(detour_key_object, type(detour_key_object)))
  
if selected_row_df:
  st.dataframe(selected_row[0])
