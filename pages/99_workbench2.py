import streamlit as st
import time

### --- TEST st.empty --- ###
if st.button('test st.empty()'):
    with st.empty():
        for seconds in range(60):
            st.write(f"⏳ {seconds} seconds have passed")
            time.sleep(1)
        st.write("✔️ 1 minute over!")


### --- TEST placeholder = st.empty --- ###
placeholder = st.empty()

# Replace the placeholder with some text:
if st.button('text'):
    placeholder.text("Hello")

if st.button('line chart'):
    # Replace the text with a chart:
    placeholder.line_chart({"data": [1, 5, 2, 6]})

if st.button('container'):
    # Replace the chart with several elements:
    with placeholder.container():
        st.write("This is one element")
        st.write("This is another")
        
if st.button('clear'):
    # Clear all those elements:
    placeholder.empty()
