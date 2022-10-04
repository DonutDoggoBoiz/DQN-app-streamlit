import streamlit as st
import datetime
from functions import fetch_price_data, observe_price, split_dataset, set_parameters, train_model, split_dataset2, test_model, save_model

### ------------ INTERFACE ------------ ###
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Select Data 📈", "Parameters 💡", "Train Model 🚀", "Test Model 🧪", "Save Model 💾"])

with tab1:
    st.header("Select stock and price range 📈")
    fetch_price_data()
    observe_button = st.checkbox('Observe Dataset 🔍')
    if observe_button:
      observe_price()
      split_button = st.checkbox("Split dataset ✂️")
      if split_button:
        st.write("Spliting.........")
        split_dataset2()

with tab2:
    st.header("Set parameters for your trading model 💡")
    set_param_button = st.checkbox("Set Parameters")
    if set_param_button:
      set_parameters()

with tab3:
    st.header("Train your model with train set 🚀")
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
    save_button = st.button("Save 💾")
    if save_button:
        #save_model()
        st.success("Your model is saved successfully")
