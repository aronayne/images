for j_rn in range(0 , 15):
    
    %reset -f

    # The length of attributes for the training examples
    hidden_size = 125
    shift_period_size = 50
    w_size = 50
    learning_rate = .001
    epsilon = .1
    epsilon_decay = .01
    gamma = .9
    number_episodes = 5
    optimizer = 'adam' # Options are 'adam' or 'SGD'
    csv_data_file = 'https://raw.githubusercontent.com/aronayne/public/master/btc-df.csv'

    results_path = '/usr/local/bin/marketagent/jupyterlab-notebooks/active-working-notebooks/RL/models/dqn-model-config.csv'

#     import matplotlib.pyplot as plt
#     import numpy as np
    from datetime import datetime
    from sklearn import metrics
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.utils.data as data_utils
    import torch.nn.functional as F
    import random
    from torch.autograd import Variable
    import pandas as pd
    import unittest
    import time
    from collections import Counter
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML
    import io
    import base64
    from ast import literal_eval
    import math 
    from datetime import date
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    pd.options.mode.chained_assignment = None
    print('1')

    ############################################################# read csv file ################################################################################# 
    print('2')
    price_data_df = pd.read_csv(csv_data_file)
    print('3')
    ############################################################# Format data into correct type for processing ################################################################################# 

    price_data_df = price_data_df.dropna()
    def convert(x):
        x = x.replace('[' , '').replace(']' , '').replace(',' , '').split(" ")  
        a = []
        for c in x : 
            if len(c) != 0 :
                a.append(float(c))

        return a

    price_data_df['PriceIntervalL1Norm'] = price_data_df['PriceIntervalL1Norm'].apply(lambda x: convert(x))
    price_data_df['TestPriceNorm'] = price_data_df['TestPriceNorm'].apply(lambda x: convert(x))

    model_config_dataset_read = None
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    shift_period_size_config = np.array([shift_period_size])
    dqn_input_size_config = np.array([hidden_size])
    w_size_config = np.array([w_size])
    learning_rate_config = np.array([learning_rate])
    epsilon_config = np.array([epsilon])
    gamma_config = np.array([gamma])
    number_episodes_config = np.array([number_episodes])
    model_config_dataset = pd.DataFrame()
    model_config_dataset['PeriodShiftSize'] = shift_period_size_config.tolist()
    model_config_dataset['dqn_input_size_config'] = dqn_input_size_config.tolist()
    model_config_dataset['w_size'] = w_size_config.tolist()
    model_config_dataset['learning_rate'] = learning_rate_config.tolist()
    model_config_dataset['epsilon'] = epsilon_config.tolist()
    model_config_dataset['gamma'] = gamma_config.tolist()
    model_config_dataset['execution_time'] = dt_string
    model_config_dataset['number_episodes'] = number_episodes
    model_config_dataset['optimizer'] = optimizer
    model_config_dataset['loss_values'] = None
    model_config_dataset['all_reward'] = None
    model_config_dataset['std_predictions_training'] = None
    model_config_dataset['std_predictions_test'] = None
    model_config_dataset['model_actions_training'] = None
    model_config_dataset['model_actions_test'] = None
    model_config_dataset['e_decay_values'] = None
    model_config_dataset['start_time'] = None
    model_config_dataset['end_time'] = None
    model_config_dataset['number_epochs'] = None
    model_config_dataset['training_set_actions_count'] = None
    model_config_dataset['test_set_actions_count'] = None
    model_config_dataset['total_reward_sum_episodes'] = None
    model_config_dataset['model_actions_training'] = model_config_dataset['model_actions_training'].astype(object)
    model_config_dataset['model_actions_test'] = model_config_dataset['model_actions_test'].astype(object)
    model_config_dataset['loss_values'] = model_config_dataset['loss_values'].astype(object)
    model_config_dataset['all_reward'] = model_config_dataset['all_reward'].astype(object)
    model_config_dataset['std_predictions_training'] = model_config_dataset['std_predictions_training'].astype(object)
    model_config_dataset['std_predictions_test'] = model_config_dataset['std_predictions_test'].astype(object)
    model_config_dataset['e_decay_values'] = model_config_dataset['e_decay_values'].astype(object)
    model_config_dataset['start_time'] = model_config_dataset['start_time'].astype(object)
    model_config_dataset['end_time'] = model_config_dataset['end_time'].astype(object)
    model_config_dataset['number_epochs'] = model_config_dataset['number_epochs'].astype(object)
    model_config_dataset['training_set_actions_count'] = model_config_dataset['training_set_actions_count'].astype(object)
    model_config_dataset['test_set_actions_count'] = model_config_dataset['test_set_actions_count'].astype(object)
    model_config_dataset['total_reward_sum_episodes'] = model_config_dataset['total_reward_sum_episodes'].astype(object)


    if not Path(results_path).is_file() or os.path.getsize(results_path) == 0 :
        print('Writing new data')
        model_config_dataset.to_csv(results_path)
    else :
        print('Appending data')
        model_config_dataset_read = pd.read_csv(results_path, index_col=0)
        model_config_dataset_read = model_config_dataset_read.append(model_config_dataset , sort=False)
        model_config_dataset_read.to_csv(results_path)
    model_config_dataset_read = pd.read_csv(results_path, index_col=0)

    shift_period_size = int(model_config_dataset.iloc[0]['PeriodShiftSize'])
    hidden_size = int(model_config_dataset.iloc[0]['dqn_input_size_config'])
    w_size = int(model_config_dataset.iloc[0]['w_size'])
    learning_rate =float( model_config_dataset.iloc[0]['learning_rate'])
    epsilon = float(model_config_dataset.iloc[0]['epsilon'])
    gamma = float(model_config_dataset.iloc[0]['gamma'])
    number_episodes = int(model_config_dataset.iloc[0]['number_episodes'])
    # model_config_dataset_read.tail()
    dataset = price_data_df
    dataset = dataset[:200]

    print('len(btc_df)' , len(price_data_df))
    print('len model_config_dataset_read' , len(model_config_dataset_read))
    print('Dataframe downloaded')
    print(sum(price_data_df['mean_price_for_interval']))
    print(len(price_data_df))
    print('Begin ' , dataset.head(1)['TimeInterval'].iloc[0][1:58])
    print('End ' , dataset.tail(1)['TimeInterval'].iloc[0][1:58])
    print(model_config_dataset.head())

    training_column = 'PriceIntervalL1Norm'

    rewards = []
    for i in range(1 , shift_period_size):
        rewards.append([s for s in dataset['reward-buy_t_'+str(i)].to_numpy()])
        rewards.append([s for s in dataset['reward-hold_t_'+str(i)].to_numpy()])
        rewards.append([s for s in dataset['reward-sell_t_'+str(i)].to_numpy()])

    actions = []
    for i in range(0 , len(rewards)):
        actions.append(i)
    input_size = len(dataset.iloc[0][training_column])
    num_classes = len(actions)

    # Model Config
    d_len = len(dataset)
    training = dataset[:int(d_len * .8)]
    test = dataset[:int(d_len * .2)]
    training_set_len = len(training[training_column])
    ############################################################# Neural Network ################################################################################# 
    states = [s for s in training[training_column].to_numpy()]
    x = torch.tensor(states , requires_grad=True).float()
    device = 'cpu'
#     torch.manual_seed(24) # <> ensure any randomness is deterministic for separate model executions in order to produce reproducable results
    class NeuralNet(nn.Module) :     
        def __init__(self, input_size, hidden_size, num_classes, lr) :  
            super(NeuralNet, self).__init__()
            self.criterion = torch.nn.MSELoss()
            print('input_size' , input_size)
            print('learning rate' , lr)
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(input_size, 400), # <> input_size is number of actions available to the RL agent. 
                            torch.nn.ReLU(),
                            torch.nn.Linear(400, 300),
                            torch.nn.ReLU(),
                            torch.nn.Linear(300, 250),
                            torch.nn.ReLU(),
                            nn.Dropout(p=0.2),
                            torch.nn.Linear(250, num_classes)
    #                         torch.nn.ReLU(),
    #                         torch.nn.Sigmoid(),
    #                         torch.nn.Dropout(),
    #                         torch.nn.Linear(250, num_classes)
                    )
            if optimizer == 'SGD' : 
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr)
            elif optimizer == 'adam' : 
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
            else : 
                raise Exception("Optimizer option not configured correctly") 

        def update(self, state, action):
            y_pred = self.model(Variable(torch.Tensor(state)))
            loss = self.criterion(y_pred, Variable(torch.Tensor(action)))
            self.optimizer.zero_grad()
            loss.backward() # <> back propogate the prediction errors to neural network weights
            self.optimizer.step() 
            return loss

        def predict(self, s):
            with torch.no_grad():
                return self.model(torch.Tensor(s))

    def gen_epsilon_greedy_policy(model, epsilon): # <> Either choose a random action from the set of available actions or predict the action from policy function approximator
        def policy_function(state):
            if random.random() < epsilon:
                return random.choice(actions)
            else:
                q_values = model.predict(state)
                return torch.argmax(q_values).item() # <> Return the action that contains the max q value for the given state
        return policy_function

    episode_reward = []
    total_episode_reward = []
    states_actions_reward_per_episode = []
    std_predictions_training = []
    std_predictions_test = []

    def weights_init(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 1) # <> Intialise the weights to follow a normal distirbution

    rewards_mapping = [] # <> Stores the reward for each action in the training set
    for i in range(1 , w_size):
        rewards_mapping.append([('reward-buy_t_'+str(i) , s) for s in training['reward-buy_t_'+str(i)].to_numpy()]) # <> Determine if a win has occured by a diaganol match
        rewards_mapping.append([('reward-hold_t_'+str(i) ,s) for s in training['reward-hold_t_'+str(i)].to_numpy()])
        rewards_mapping.append([('reward-sell_t_'+str(i) ,s) for s in training['reward-sell_t_'+str(i)].to_numpy()])
    rewards_mapping_test = [] # <> Stores the reward for each action in the test set
    for i in range(1 , w_size):
        rewards_mapping_test.append([('reward-buy_t_'+str(i) , s) for s in test['reward-buy_t_'+str(i)].to_numpy()])
        rewards_mapping_test.append([('reward-hold_t_'+str(i) ,s) for s in test['reward-hold_t_'+str(i)].to_numpy()])
        rewards_mapping_test.append([('reward-sell_t_'+str(i) ,s) for s in test['reward-sell_t_'+str(i)].to_numpy()])
    ############################################################# Execute Model ################################################################################# 
    model = NeuralNet(input_size, hidden_size, num_classes, learning_rate) # <> Create an instance of the policy function approximator
    model.apply(weights_init)
    loss_values = [] # <> Store the MSE loss of the policy function approximator
    e_decay_values = []
    counter = 0
    total_reward = 0
    reward = 0
    all_reward = []
    epoch = 0
    start_time =  int(round(time.time() * 1000))
    end_time_last_epoch =  None
    epsilon_decay_values = []

    for episode in range(number_episodes):
        i = 0
        policy = gen_epsilon_greedy_policy(model, epsilon)
        while i < training_set_len-1:
            state = x[i] # <> Access the state indexed by i 
            action = policy(state) # <> For the policy approximation determine the action for the state
            next_state = x[i+1]
            reward = rewards[action][i]
            q_values = model.predict(state) # <> Determine Q values for state in iteration i

            q_values_next = model.predict(next_state) # <> Store the MSE loss of the policy function approximator
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            loss_value = model.update(state, q_values_next)
            loss_values.append(loss_value)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            i = i + 1
    ############################################################ Logging ################################################################################ 
            states_actions_reward_per_episode.append((state , action , reward , epsilon))
            total_reward = total_reward + reward
            episode_reward.append(total_reward)
            counter = counter + 1
            all_reward.append(total_reward)
            model_actions_training = [int(torch.argmax(model.predict(a))) for a in training[training_column].to_numpy()]

            total_reward_training = 0
            ii = 0
            for iii in range(len(model_actions_training)) : 
                total_reward_training = total_reward_training + rewards_mapping[model_actions_training[iii]][ii][1] # <> on the training set measure the total amount reward gained if were to select actions using the current state action mappings
                ii = ii + 1
            model_actions_test = [int(torch.argmax(model.predict(a))) for a in test[training_column].to_numpy()]

            total_reward_test = 0
            ii = 0
            for iii in range(len(model_actions_test)) : 
                total_reward_test = total_reward_test + rewards_mapping[model_actions_test[iii]][ii][1] # <> on the test set measure the total amount reward gained if were to select actions using the current state action mappings
                ii = ii + 1

            if counter % 100 == 0:
                print(counter , 'std training actions:' , np.std(model_actions_training), 'std test actions:' , np.std(model_actions_test)) # <> On the training set measure the standard deviation of the model actions 
                print('total_reward training:' , total_reward_training, 'total_reward test:' , total_reward_test)

            std_predictions_training.append(np.std(model_actions_training))  
            std_predictions_test.append(np.std(model_actions_test))   

            epoch = epoch + 1
            end_time_last_epoch =  int(round(time.time() * 1000))

        epsilon = epsilon - 0.01 # <> Decay epsilon every 3 episodes
        e_decay_values.append(epsilon)

        print('episode: {}'.format(episode)) # <> Decay epsilon for each episode

        total_episode_reward.append(total_reward)

    print('rewards_mapping shape(' , len(rewards_mapping) ,',' , len(rewards_mapping[0]) , ')')
    print('len(actions)' , len(actions))
    print('Training set instances count: ' , len(training))
    print('Test set instances count' , len(test))

    ############################################################ Write results to file ################################################################################ 

    model_config_dataset_read = pd.read_csv(results_path, index_col=0)
    model_config_dataset_read['loss_values'] = model_config_dataset_read['loss_values'].astype(object)
    model_config_dataset_read['all_reward'] = model_config_dataset_read['all_reward'].astype(object)
    model_config_dataset_read['std_predictions_training'] = model_config_dataset_read['std_predictions_training'].astype(object)
    model_config_dataset_read['std_predictions_test'] = model_config_dataset_read['std_predictions_test'].astype(object)
    model_config_dataset_read['model_actions_training'] = model_config_dataset_read['model_actions_training'].astype(object)
    model_config_dataset_read['model_actions_test'] = model_config_dataset_read['model_actions_test'].astype(object)
    model_config_dataset_read['e_decay_values'] = model_config_dataset_read['e_decay_values'].astype(object)
    model_config_dataset_read['total_reward_sum_episodes'] = model_config_dataset_read['total_reward_sum_episodes'].astype(object)
    model_config_dataset_read['loss_values'].iloc[len(model_config_dataset_read) - 1] = [float(l) for l in loss_values]
    model_config_dataset_read['all_reward'].iloc[len(model_config_dataset_read) - 1] = all_reward
    model_config_dataset_read['std_predictions_training'].iloc[len(model_config_dataset_read) - 1] = std_predictions_training
    model_config_dataset_read['std_predictions_test'].iloc[len(model_config_dataset_read) - 1] = std_predictions_test
    # e_decay_values = [e[3] for e in states_actions_reward_per_episode]
    model_config_dataset_read['e_decay_values'].iloc[len(model_config_dataset_read) - 1] = e_decay_values
    model_config_dataset_read['number_epochs'].iloc[len(model_config_dataset_read) - 1] = number_episodes
    model_config_dataset_read['start_time'].iloc[len(model_config_dataset_read) - 1] = start_time
    model_config_dataset_read['end_time'].iloc[len(model_config_dataset_read) - 1] = end_time_last_epoch
    model_config_dataset_read['model_actions_training'].iloc[len(model_config_dataset_read) - 1] = model_actions_training
    model_config_dataset_read['model_actions_test'].iloc[len(model_config_dataset_read) - 1] = model_actions_test
    model_config_dataset_read['training_set_actions_count'].iloc[len(model_config_dataset_read) - 1] = json.dumps(Counter(model_actions_training))
    model_config_dataset_read['test_set_actions_count'].iloc[len(model_config_dataset_read) - 1] = json.dumps(Counter(model_actions_test))
    model_config_dataset_read['total_reward_sum_episodes'].iloc[len(model_config_dataset_read) - 1] = sum(total_episode_reward)

    # print(e_decay_values)
    model_config_dataset_read.to_csv(results_path)
    model_config_dataset_read['start_time'] = pd.to_datetime(model_config_dataset_read['start_time'], unit='ms')
    model_config_dataset_read['end_time'] = pd.to_datetime(model_config_dataset_read['end_time'], unit='ms')
