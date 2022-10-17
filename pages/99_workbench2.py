### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time

from functions import fetch_price_data, observe_price, split_dataset2, set_parameters, set_train_episodes, train_model, test_model, save_model

### --- DATABASE CONNECTION --- ###
deta = Deta(st.secrets["deta_key"])
user_db = deta.Base("user_db")

user_frame = pd.DataFrame(user_db.fetch().items)
user_list = user_frame['username'].values.tolist()
password_list = user_frame['password'].values.tolist()

### --- SESSION STATE --- ###
if 'login_status' not in st.session_state:
  st.session_state['login_status'] = False
if 'username' not in st.session_state:
  st.session_state['username'] = None
if 'col3_b_status' not is st.session_state:
  st.session_state['col3_b_status'] = False
  
def login_func():
  st.session_state['login_status'] = True

def logout_func():
  st.session_state['login_status'] = False
  st.session_state['username'] = None
  
def rerun():
  st.experimental_rerun()
  
### --- INTERFACE --- ###
placeholder = st.empty()

if st.session_state['login_status'] == False:
    with placeholder.container():
        login_form = st.form('Login')
        login_form.subheader('Login 📝')
        username = login_form.text_input('Username', placeholder='your username')
        password = login_form.text_input('Password', type='password', placeholder='your password')
        if login_form.form_submit_button('Login'):
            if len(username) <= 0:
              st.warning("Please enter a username")
            elif len(password) <= 0:
              st.warning("Please enter your password")
            elif len(username) > 0 and len(password) > 0:
              if username not in user_list:
                st.warning('User not found. Please check your username')
              else: 
                if user_frame.loc[user_frame['username'] == username,'password'].values != password:
                  st.error("Password incorrect. Please try again")
                else:
                  st.success("Login Successful!")
                  st.session_state['username'] = username
                  login_func()
                  time.sleep(2)
                  rerun()
                  #with placeholder.container():
                    #st.write('#### Welcome, {}'.format(st.session_state['username']))
                    #st.button("Let's start!")
                  ### --- SIDEBAR --- ###
                  #st.sidebar.write('Welcome, {}'.format(st.session_state['username']))
                  #st.sidebar.button('Logout', on_click=logout_func)
                  #st.sidebar.button('Reset Password')
else:
    st.sidebar.write('Welcome, {}'.format(st.session_state['username']))
    ### --- SIDEBAR --- ###
    logout_button_side = st.sidebar.button('Logout', on_click=logout_func)
    reset_pass_button_side = st.sidebar.button('Reset Password')
    
    ### --- MAIN --- ###
    col1, col2, col3 = st.columns(3)
    with col1:
      col1_b = st.button('col 1')
    with col2:
      col2_b = st.button('col 2')
    with col3:
      col3_b = st.button('col 3 GO')
    
    with placeholder.container():
        st.write('### Welcome, {}'.format(st.session_state['username']))
      
    if col3_b or st.session_state['col3_b_status']:  
      st.session_state['col3_b_status'] = True
      with placeholder.container():
        st.write('proceed')
        ### ------------ INTERFACE ------------ ###
        tab_list = ["Select Data 📈", "Set Parameters 💡", "Train Model 🚀", "Test Model 🧪", "Save Model 💾", "PENDING"]
        select_data_tab, set_para_tab, train_tab, test_tab, save_tab, pending_tab = st.tabs(tab_list)

        with select_data_tab:
            st.header("Select stock and price range 📈")
            fetch_price_data()
            observe_button = st.checkbox('View Dataset 🔍')
            if observe_button:
              observe_price()
              split_button = st.checkbox("Split dataset ✂️")
              if split_button:
                split_dataset2()

        with set_para_tab:
            st.header("Set parameters for your trading model 💡")
            #set_param_button = st.checkbox("Set Parameters")
            #if set_param_button:
              #set_parameters()
            set_parameters()

        with train_tab:
            st.header("Train your model with train set 🚀")
            col1 , col2 = st.columns(2)
            with col1:
                set_train_episodes()
            with col2:
                st.write('  ')
                st.write('  ')
                train_button = st.checkbox("Start Training 🏃")
            if train_button:
              train_model()


        with test_tab:
            st.header("Test your model on test set 🧪")
            test_button = st.checkbox("Start Testing 🏹")
            if test_button:
                st.write("Test Result")
                test_model()

        with save_tab:
            st.header("Save your model")
            show_model_list_checkbox = st.checkbox('Show model list')
            if show_model_list_checkbox:
              st.write(model_df)
            save_button = st.button("Save 💾")
            if save_button:
                save_model()
                st.success("Your model is saved successfully")

        with pending_tab:
            st.header("PENDING adjustment...")
            st.success("select data ---- DONE")
            st.warning("parameter -- adjust interface and input choice")
            st.warning("parameter -- add info to each input")
            st.success("train model -- add input field for n_episodes ---- DONE")
            st.warning("train / test model -- better result visualization")
            st.warning("save model -- integrate to cloud infrastructure")
            st.warning("generate advice -- add load_model function")
            st.warning("generate advice -- compulsory stock quote")
            st.warning("generate advice -- formally written buy/sell advice")
            st.error("overall -- user database and management system")
            st.error("overall -- stock quote database")
            st.error("overall -- set up cloud infrastructure")


