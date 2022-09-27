import streamlit as st

if st.checkbox("Test sess_momo"):
  st.session_state['sess_momo'] = 9999
  sess_momo = st.session_state['sess_momo']
  st.write("start sess_momo = {} ".format(st.session_state['sess_momo']))
  momo_paradise = 99

if st.checkbox("Test normal momo"):
  momo_paradise = 11
  st.write("start momo = {} ".format(momo_paradise) )

if st.checkbox("End momo"):
  st.write('end momo = {}'.format(momo_paradise) )
  st.write('end sess_momo = {}'.format(sess_momo) )
