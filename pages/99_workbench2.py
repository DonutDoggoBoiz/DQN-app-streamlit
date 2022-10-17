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
# BUTTONS
test_b =st.button('text'):
line_chart_b = st.button('line chart'):
container_b = st.button('container'):       
clear_b = st.button('clear'):

# SINGLE ELEMENT CONTAINER
placeholder = st.empty()

# BUTTON CONDITIONS
# Replace the placeholder with some text:
if test_b:
    placeholder.text("Hello")

if line_chart_b:
    # Replace the text with a chart:
    placeholder.line_chart({"data": [1, 5, 2, 6]})

if container_b:
    # Replace the chart with several elements:
    with placeholder.container():
        st.write("This is one element")
        st.write("This is another")
        
if clear_b:
    # Clear all those elements:
    placeholder.empty()
    

