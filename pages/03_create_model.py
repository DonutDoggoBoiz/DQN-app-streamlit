import streamlit as st

# st.markdown("# Create MODEL 🚀")
# st.sidebar.markdown("# Create Model 🚀")

st.title('Create DQN Trading Model 🚀')
st.sidebar.markdown('## Create Model 🚀')

stock_name = st.selectbox('Select your Stock', ('BBL', 'PTT', 'ADVANC','KBANK') )

x = [1,2,3,4,5,6,7,8,9,10]

for n in x:
  st.write(x)
