import streamlit as st
import altair as alt
# import matplotlib.pyplot as plt
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

observe_button = st.checkbox('Observe')

if observe_button:    
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
    st.write('Dataset will be split into {} records of train set and {} records of test set'.format(split_point, df_length-split_point) )
    st.write('considered as {:.2f}% of train set and {:.2f}% of test set'.format(train_size_pct,test_size_pct) )
    
    split_button = st.checkbox('Split dataset')
    if split_button:
      train_prices = df_price.loc[:split_point, 'Close'].to_numpy()
      test_prices = df_price.loc[split_point:, 'Close'].to_numpy()
      st.write('Train set' + df_price.loc[:split_point, 'Close'])
      st.write('-----------------------------------------------'])
      st.write('Test set' + df_price.loc[split_point:, 'Close'])
#else:
    #st.write('Click "Observe" button to observe historical price chart')
