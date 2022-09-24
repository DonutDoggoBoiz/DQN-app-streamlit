import streamlit as st
import altair as alt
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# st.markdown("# Create MODEL 🚀")
# st.sidebar.markdown("# Create Model 🚀")

st.title('Create DQN Trading Model 🚀')
st.sidebar.markdown('## Create Model 🚀')

stock_name = st.selectbox('Select your Stock', ('BBL', 'PTT', 'ADVANC','KBANK') )
start_date = st.date_input("Select start date: ", datetime.date(2021, 9, 20))
end_date = st.date_input("Select end date: ", datetime.date(2022, 9, 20))

if st.button('Observe'):
    stock_code = stock_name + '.BK'
    df_price = yf.download(stock_code,
                        start=start_date,
                        end=end_date,
                        progress=True)
    df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
    # st.line_chart(df_price['Close'])
    c = alt.Chart(df_price['Close'].reset_index()).mark_line().encode(
        x='Date:T',
        y='Price:Q')
    st.altair_chart(c)
    #st.write(df_price['Close'].reset_index())
    
else:
    st.write('Click "Observe" button to observe historical price chart')
