import streamlit as st
import pandas as pd
import datetime
from functions import fetch_price_data, observe_price, split_dataset, set_parameters, train_model, split_dataset2, test_model, save_model

### ------------ session state ------------ ###
username = 'admin99'
### ------------ MODEL DATABASE ------------ ###
model_list = []
model1 = {'username':'admin', 'model_name':'bbl_01', 'stock_quote':'bbl'.upper()}
model2 = {'username':'admin', 'model_name':'ptt_04', 'stock_quote':'ptt'.upper()}
model_list.append(model1)
model_list.append(model2)
model_df = pd.DataFrame(model_list)

### ------------ INTERFACE ------------ ###
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Select Data 📈", "Set Parameters 💡", "Train Model 🚀", "Test Model 🧪", "Save Model 💾", "PENDING"])

with tab1:
    st.header("Select stock and price range 📈")
    fetch_price_data()
    observe_button = st.checkbox('View Dataset 🔍')
    if observe_button:
      observe_price()
      split_button = st.checkbox("Split dataset ✂️")
      if split_button:
        #st.write("Spliting.........")
        split_dataset2()

with tab2:
    st.header("Set parameters for your trading model 💡")
    #set_param_button = st.checkbox("Set Parameters")
    #if set_param_button:
      #set_parameters()
    set_parameters()

with tab3:
    st.header("Train your model with train set 🚀")
    col1 , col2 = st.columns(2)
    with col1:
        train_episodes = st.number_input('Number of training episodes:', value=2, step=1, min_value=0)
    with col2:
        st.write('  ')
        st.write('  ')
        train_button = st.button("Start Training 🏃")
        if train_button:
          train_model()
        

with tab4:
    st.header("Test your model on test set 🧪")
    test_button = st.button("Start Testing 🏹")
    if test_button:
        st.write("Test Result")
        test_model()
    
with tab5:
    st.header("Save your model")
    show_model_list_checkbox = st.checkbox('Show model list')
    if show_model_list_checkbox:
      st.write(model_df)
    save_button = st.button("Save 💾")
    if save_button:
        #save_model()
        new_model = {'username':username, 'model_name':agent_name, 'stock_quote':'ptt'.upper()}
        model_df.append(new_model)
        st.success("Your model is saved successfully")
        st.write(model_df)
        
        
with tab6:
    st.header("PENDING adjustment...")
    st.success("select data = DONE")
    st.warning("parameter -- adjust interface and input choice")
    st.warning("parameter -- add info to each input")
    st.warning("train model -- add input field for n_episodes")
    st.warning("test model -- better result visualization")
    st.warning("save model -- integrate to cloud infrastructure")
    st.warning("generate advice -- add load_model function")
    st.warning("generate advice -- compulsory stock quote")
    st.warning("generate advice -- formally written buy/sell advice")
    st.error("overall -- user database and management system")
    st.error("overall -- stock quote database")
    st.error("overall -- set up cloud infrastructure")
     
             
             

