import streamlit as st

#st.markdown("### This is a Login Page 🔑")
#st.sidebar.markdown("### Login 🔑")
#st.write('--------')

# --- mocked up User Database ---
if st.sidebar.checkbox('database'):
  users = ['admin']
  passwords = ['admin']

st.title("Simple Login App")

menu = ["Home", "Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
  st.subheader("Home")
  
elif choice == "Login":
  st.subheader("Login Section")
  username = st.sidebar.text_input("Username")
  password = st.sidebar.text_input("Password", type='password')
  if st.sidebar.checkbox("Login"):
    #if password == '12345':
    if passwords[users.index(username)] == password:
      st.success("Logged In as {}".format(username))

      task = st.selectbox("Task", ["Add Post", "Analytics", "Profiles"])
      if task == "Add Post":
        st.subheader("Add Your Post")
      elif task == "Analytics":
        st.subheader("Analytics")
      elif task == "Profiles":
        st.subheader("Profiles")
        users_df = pd.DataFrame({'users':users,
                                 'passwords':passwords}
                               )
        st.dataframe(users_df)
    else:
      st.warning("Incorrect Username/Password")
  
elif choice == "Sign Up":
  st.subheader("Create New Account")
  new_user = st.text_input("Username")
  new_password = st.text_input("Password", type='password')
  
  if st.button("Signup"):
    users.append(new_user)
    passwords.append(new_password)
    st.success("You have successfully created a valid account")
    st.info("Go to Login Menu to login")
