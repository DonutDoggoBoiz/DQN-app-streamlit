# --- IMPORT LIBRARY
import numpy as np
import pandas as pd
import yfinance as yf
  # from dqn_object import Agent
import streamlit as st
# ------------------------

# --- PRICE FETCHING MODULE ---
def fetch_price_data(stock_code, start_date, end_date):
  df_price = yf.download(stock_code,
                        start=start_date,
                        end=end_date,
                        progress=True)
  df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
  

def split_dataset(split_point):
  train_prices = df_price['Close'][:split_point].to_numpy()
  test_prices = df_price['Close'][split_point:].to_numpy()
  
# --- MACHINE LEARNING MODULE ---
## --- parameters setting ---
def set_parameters():
  ### --- environment parameters
  action_space = 2      # consist of 0(Sell) , 1(Buy)
  window_size = 5      # n-days of prices used as observation or state
  n_episodes = 50      # 10ep use around 6 mins

  ### --- agent parameters
  agent_gamma = 0.99              # discount rate for Q rewards
  agent_epsilon = 1.0                 # probability of explorative actions
  agent_epsilon_dec = 0      # 1e-3  # decline of p(explore) each steps
  agent_epsilon_end = 0.01        # minimum probability of explorative actions
  agent_lr = 0.001                      # learning-rate for the optimizer in neural network

  ### --- trading parameters
  initial_balance = 1000000
  trading_size_pct = 10
  commission_fee_pct = 0.157
  trade_size = (trading_size_pct/100) * initial_balance
  commission_fee = (commission_fee_pct/100) * 1.07

  ### --- episodic History
  total_acc_reward_history = []
  end_balance_history = []
  eps_history = []

  ### --- trading History
  acc_reward_history = []
  action_history = []
  account_balance_history = []
  nom_return_history = []
  real_return_history = []

# --- TRAINING MODULE ---
## --- initialize agent object
def train_model():
  agent = Agent(
                gamma=agent_gamma, 
                epsilon=agent_epsilon, 
                epsilon_dec=agent_epsilon_dec,
                lr=agent_lr, 
                input_dims=window_size,
                n_actions=action_space, 
                mem_size=1000000, 
                batch_size=32,
                epsilon_end=agent_epsilon_end)

  ## --- loop through episodes
  for i in range(n_episodes):
      ### --- start episode --- ###
      print ("---------- Episode " + str(i+1) + " / " + str(n_episodes) + ' ----------' )

      # slider window
      start_tick = window_size
      end_tick = len(train_prices) - 2 
      current_tick = start_tick
      ##last_trade_tick = current_tick - 1
      done = False

      # bundle train_prices data into state and new_state
      state = train_prices[ (current_tick - window_size) : current_tick ]
      new_state = train_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

      # initiate episodial variables
      acc_reward = 0
      account_balance = initial_balance
      trade_exposure = False
      trade_exposure_ledger = []
      last_buy = []

      while not done:
        action = agent.choose_action(state)

        if action == 1: # buy
            reward = train_prices[current_tick+1] - train_prices[current_tick]
            acc_reward += reward
            if trade_exposure == False:
              last_buy.append(train_prices[current_tick])     # memorize bought price
              account_balance -= trade_size * commission_fee  # pay fees on purchase
              trade_exposure = True 

        elif action == 0: # sell
            reward = train_prices[current_tick] - train_prices[current_tick+1]
            acc_reward += reward
            if trade_exposure == True:
              return_pct = (train_prices[current_tick] - last_buy[-1]) / last_buy[-1]   # profit/loss percentage on investment
              market_value = (return_pct+1) * trade_size                                # market value of investment
              nom_return = return_pct * trade_size
              real_return = (return_pct * trade_size) - (market_value * commission_fee)
              account_balance += real_return
              nom_return_history.append([int(current_tick),nom_return])
              real_return_history.append([int(current_tick),real_return])
              trade_exposure = False

        done = True if current_tick == end_tick else False

        agent.store_transition(state, action, reward, new_state, done)
        agent.learn()

        current_tick += 1
        state = new_state
        new_state = train_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

        # append history lists
        acc_reward_history.append(acc_reward)
        action_history.append(action)
        account_balance_history.append(account_balance)

        if done: 
          # print ("-----------------------------------------")
          print ("Total Reward: {:.2f} , Account_Balance: {:2f}".format(acc_reward, account_balance) )
          print ("-----------------------------------------")
        ### --- end of 1 episode --- ###

      total_acc_reward_history.append(acc_reward)
      end_balance_history.append(account_balance)
      eps_history.append(agent.epsilon)

  # --- reshape history data to array ---
  record_num = np.array(action_history).shape[0]
  np_acc_reward_history = np.reshape( np.array(acc_reward_history) , ( int(n_episodes) , int(record_num/n_episodes) ) )
  np_action_history = np.reshape( np.array(action_history) , ( int(n_episodes) , int(record_num/n_episodes) ) )
  np_account_balance_history = np.reshape( np.array(account_balance_history) , ( int(n_episodes) , int(record_num/n_episodes) ) )
  # --- print shape of history arrays ---
  print('np_acc_reward_history.shape: {}'.format(np_acc_reward_history.shape))
  print('np_action_history.shape: {}'.format(np_action_history.shape))
  print('np_account_balance_history.shape: {}'.format(np_account_balance_history.shape))

# plot last 10 episodes of acc_reward_history
def last10_history():  # ********
  for i in range(n_episodes-10,n_episodes):
    pd.DataFrame(np_acc_reward_history[i]).plot(figsize=(6,3), title='episode'+str(i+1), legend=False)
