import streamlit as st
import altair as alt
# import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# st.markdown("# Create MODEL 🚀")
# st.sidebar.markdown("# Create Model 🚀")

st.title('Create DQN Trading Model 💡')
st.sidebar.markdown('## Create Model 💡')

set_param_button = st.button("Set Parameters")
if set_param_button:
        st.write("#### Create a form of widgets to set parameters")
        st.write("Setting parameters .....")
        #set_parameters()
        #st.write("action_space: {}".format(action_space) ) 
        #st.write("window_size: {}".format(window_size) ) 
        #st.write("n_episode: {}".format(n_episodes) )
        st.write("Setting parameters A")
        st.write("Setting parameters B")
        st.write("Setting parameters C")
        '''
        --- environment parameters
        action_space = 2
        window_size = 5
        n_episodes = 2

        --- agent parameters
        agent_gamma = 0.99 
        agent_epsilon = 1.0
        agent_epsilon_dec = 0
        agent_epsilon_end = 0.01
        agent_lr = 0.001

        --- trading parameters
        initial_balance = 1000000
        trading_size_pct = 10
        commission_fee_pct = 0.157
        trade_size = (trading_size_pct/100) * initial_balance
        commission_fee = (commission_fee_pct/100) * 1.07

        --- episodic History
        total_acc_reward_history = []
        end_balance_history = []
        eps_history = []

        --- trading History
        acc_reward_history = []
        action_history = []
        account_balance_history = []
        nom_return_history = []
        real_return_history = []
        '''
