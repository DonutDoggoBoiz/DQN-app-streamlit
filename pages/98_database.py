import streamlit as st
import pandas as pd
from deta import Deta

deta = Deta(st.secrets["deta_key"])
db = deta.Base("user_db")

register_form = st.form('Register')
register_form.subheader('Registration Form')
new_username = register_form.text_input('Username').lower()
new_password = register_form.text_input('Password', type='password')

if register_form.form_submit_button('Register'):
  if len(new_username) <= 0:
    st.warning("Please enter a username")
  elif len(new_password) <= 0:
    st.warning("Please enter your password")
  elif len(new_username) > 0 and len(new_password) > 0:
    db.put({'username':new_username, 'password':new_password})
    st.success("Register Successful!")

if st.button('show user database'):
  st.write('Here is the latest user database')
  aaa = db.fetch().items
  st.write(type(aaa))
  st.write(aaa)
  daframe = pd.DataFrame(aaa)
  st.write(daframe)
  #st.dataframe(db.fetch().items)

db2 = deta.Base("model_db")
