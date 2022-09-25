import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
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
stock_code = stock_name + '.BK'
df_price = yf.download(stock_code,
                        start=start_date,
                        end=end_date,
                        progress=True)
df_price.drop(columns=['Adj Close','Volume'] , inplace=True)

if st.button('Observe'):    
    #alt chart with scale
    c = (alt.Chart(df_price['Close'].reset_index()
                  )
            .mark_line()
            .encode(x = alt.X('Date') ,
                    y = alt.Y('Close', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                    tooltip=['Date','Close']
                   )
            .interactive()
        )

    st.altair_chart(c, use_container_width=True)
    
    # st.line_chart(df_price['Close'])
    df_length = df_price['Close'].count()
    st.write('This dataset contains ' + str(df_length) + ' days of historical prices')

    split_point = st.slider('Select the split point between Train set and Test set:', 0, int(df_length), int(df_length/2))
    train_size_pct = (split_point/df_length)*100
    test_size_pct = 100-train_size_pct
    st.write('The split point is ' + str(split_point) )
    st.write('It split data into ' + str(train_size_pct)+'%'+' of train set and '+ str(test_size_pct)+'%'+' of test set' )
else:
    st.write('Click "Observe" button to observe historical price chart')
