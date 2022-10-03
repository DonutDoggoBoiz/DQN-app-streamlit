import streamlit as st
import datetime
from functions import fetch_price_data, observe_price, split_dataset, set_parameters, train_model
  
### ------------ INTERFACE ------------ ###
tab1, tab2, tab3, tab4 = st.tabs(["Select Data 📈", "Parameters 💡", "Train Model 🚀", "Test Model 🧪"])

with tab1:
    st.header("Select stock and price range 📈")
    fetch_price_data()
    observe_button = st.checkbox('Observe Dataset 🔍')
    if observe_button:
      observe_price()
      split_button = st.checkbox("Split dataset ✂️")
      if split_button:
        st.write("Spliting.........")
        split_dataset()
        st.write("Train dataset")
        st.line_chart(df_price_train)
        st.write("Test dataset")
        st.line_chart(df_price_test)
        st.write("Spliting......... DONE!")
        set_param_button = st.checkbox("Set Parameters")

with tab2:
    st.header("Set parameters for your trading model 💡")
    set_param_button = st.checkbox("Set Parameters")
    if set_param_button:
      set_parameters()

with tab3:
    st.header("Train your model 🚀")
    train_button = st.button("Start Training 🏃")
    if train_button:
      train_model()

with tab4:
    st.header("Test Model on test set")
