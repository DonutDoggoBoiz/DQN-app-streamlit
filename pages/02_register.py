import streamlit as st

st.markdown("## Register Page 📝")
st.sidebar.markdown("## Register 📝")

register_user_form = st.form('Register user')
register_user_form.subheader('Registration Form')
new_username = register_user_form.text_input('Username').lower()
new_password = register_user_form.text_input('Password', type='password')
# new_password_repeat = register_user_form.text_input('Repeat password', type='password')

if register_user_form.form_submit_button('Register'):
  if len(new_username) <= 0:
    st.warning("Please enter a username")
  elif len(new_password) <= 0:
    st.warning("Please enter your password")
  elif len(new_username) > 0 and len(new_password) > 0:
    st.success("Register Successful!")
