import streamlit as st

st.markdown("# This is a Login Page 🔑")
st.sidebar.markdown("# Login 🔑")

"""
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]
passwords = ["abc123", "def456"]

authenticator = stauth.Authenticate(names, usernames, passwords,
                                    "sales_dashboard", "abcdef", cookie_expiry_days=0)
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
  st.error("Username/password is incorrect")

if authentication_status == None:
  st.warning("Please enter your username and password")

if authentication_status:
  st.write("### Login Success")
  st.write("### Welcome to our app!")
  
  # --- SIDEBAR ---
  authenticator.logout("Logout", "sidebar")
  st.sidebar.title(f"Welcome {name}")
"""

st.write('----None----')
