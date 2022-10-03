import streamlit as st
  
### ------------ INTERFACE ------------ ###
tab1, tab2, tab3, tab4 = st.tabs(["Select Data", "Parameters", "Train Model", "Test Model"])

with tab1:
    st.header("Select stock and price range")
    stock_name = st.selectbox('Select your Stock', ('BBL', 'PTT', 'ADVANC','KBANK') )
    start_date = st.date_input("Select start date: ", datetime.date(2021, 9, 20))
    end_date = st.date_input("Select end date: ", datetime.date(2022, 9, 20))
    stock_code = stock_name + '.BK'
    df_price = yf.download(stock_code,
                            start=start_date,
                            end=end_date,
                            progress=True)
    df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
    df_length = df_price['Close'].count()

with tab2:
   st.header("Set parameters for your trading model")
   

with tab3:
   st.header("Train your model")
   
