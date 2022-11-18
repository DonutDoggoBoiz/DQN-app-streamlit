### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
##############################
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
##############################
##############################
class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):  # to store experience into memory
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):     # to randomly select experience from memory
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)  # randomly select an array of 'batch_size' indexes from 'max_mem' indexes pool

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

##############################
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):   # this function will be call when we want to create a neural network
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation='linear')])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

##############################
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end  ### ***change word to eps_min
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:  # exploration
            action = np.random.choice(self.action_space)
        else:  # exploitation
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self): 
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)

        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min  # 1.00 ---> eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

##############################
def fetch_price_data():
  global df_price, df_length
  stock_name = st.selectbox('Select your Stock', ('BBL', 'PTT', 'ADVANC','KBANK'))
  start_date = st.date_input("Select start date: ", datetime.date(2021, 5, 18))
  end_date = st.date_input("Select end date: ", datetime.date(2022, 11, 18))
  stock_code = stock_name + '.BK'
  df_price = yf.download(stock_code,
                        start=start_date,
                        end=end_date,
                        progress=True)
  df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
  df_length = len(df_price['Close'])
  
def observe_price():
  c = alt.Chart(df_price['Close'].reset_index()).encode(
      x = alt.X('Date'),
      y = alt.Y('Close',
                scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10])),
      tooltip=['Date','Close'])
  st.altair_chart(c.mark_line().interactive(), use_container_width=True)
  st.write('This dataset contains ' + str(df_length) + ' days of historical prices')
  global split_point
  split_point = int(df_length/2)
  train_size_pct = (split_point/df_length)*100
  test_size_pct = 100-train_size_pct
  st.write('Dataset will be split into {} records of train set and {} records of test set'.format(split_point, df_length-split_point) )
  st.write('train set will be considered as {:.2f}% of dataset while the other {:.2f}% is test set'.format(train_size_pct,test_size_pct) )
  
##############################
def split_dataset():
  global df_price_train, df_price_test, train_prices, test_prices
  df_price_train = df_price['Close'][:split_point]
  df_price_test = df_price['Close'][split_point:]
  train_prices = df_price['Close'][:split_point].to_numpy()
  test_prices = df_price['Close'][split_point:].to_numpy()

##############################
def set_parameters():
  ### --- environment parameters
  action_space = 2
  window_size = 5
  n_episodes = 2

  ### --- agent parameters
  agent_gamma = 0.99
  agent_epsilon = 1.0
  agent_epsilon_dec = 0
  agent_epsilon_end = 0.01
  agent_lr = 0.001

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
  

##############################
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
      st.write("---------- Episode " + str(i+1) + " / " + str(n_episodes) + ' ----------' )

      # slider window
      start_tick = window_size
      end_tick = len(train_prices) - 2 
      current_tick = start_tick
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
          #print ("Total Reward: {:.2f} , Account_Balance: {:2f}".format(acc_reward, account_balance) )
          #print ("-----------------------------------------")
          st.write("------------- Episode {} of {} done...".format(i+1, n_episodes) )
          st.write("-------------Total Reward: {:.2f} , Account_Balance: {:2f}".format(acc_reward, account_balance) )
        ### --- end of 1 episode --- ###
          total_acc_reward_history.append(acc_reward)
          end_balance_history.append(account_balance)
          eps_history.append(agent.epsilon)
          
##############################
##############################
# streamlit_app.py

#import streamlit as st
#from google.oauth2 import service_account
#from google.cloud import storage

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)
path_uri = 'gs://streamlitapphost.appspot.com/gcs_mnist_test.csv'
bucket_name = "streamlitapphost.appspot.com"
bucket = client.bucket(bucket_name)

file_path = 'gcs_mnist_test.csv'

gcs_switch = st.checkbox('GCS Switch')
if gcs_switch:
  local_path = 'models/gcs_mnist_test.csv'
  content = bucket.blob(file_path).download_to_filename(local_path)

show_file = st.button('Show local file')
save_file = st.button('Save to GCS')

if show_file:
  try:
    local_df = pd.read_csv(local_path)
    st.write(local_df.shape)
    st.dataframe(local_df.iloc[:10,:10])
  except:
    st.error('ERROR LOCAL')
    
if save_file:
  try:
    csv_path = 'models/df_100.csv'
    local_df = pd.read_csv(local_path)
    local_df.iloc[:20,:5].to_csv(csv_path)
    gcs_path = 'csv_blob/gcs_df_100.csv'
    gcs_blob = bucket.blob(gcs_path)
    gcs_blob.upload_from_filename(csv_path)
    st.success('UPLOAD DONE!')
  except:
    st.error('ERROR UPLOAD GCS')
                               
##############################

st.write('TEST TRAIN TO SAVE')
fetch_price_button = st.checkbox('fetch price')
observe_button = st.checkbox('observe price')
split_data = st.checkbox('split price')
set_param_button = st.checkbox('set param')
train_button = st.checkbox('train train')

if fetch_price_button:
  fetch_price_data()
                         
if observe_button:
  observe_price()
                         
if split_data:
  split_dataset()
                         
if set_param_button:
  set_parameters()
                         
if train_button:
  train_model()
                               
                               
                               
                               
