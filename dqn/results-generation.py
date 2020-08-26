%reset -f

import pandas as pd
last_n_results = 50
results_path = '/usr/local/bin/marketagent/jupyterlab-notebooks/active-working-notebooks/RL/models/dqn-model-config.csv'
model_config_dataset_read = pd.read_csv(results_path, index_col=0)
model_config_dataset_read_d = model_config_dataset_read.drop(columns=['e_decay_values' , 'w_size' , 
                                                                      'execution_time', 'loss_values', 'std_predictions_test', 
                                                                      'model_actions_training', 'model_actions_test' , 'start_time',
                                                                      'end_time' , 'training_set_actions_count' , 'test_set_actions_count',
                                                                      'all_reward' , 'PeriodShiftSize' , 'dqn_input_size_config',
                                                                     'std_predictions_training'])
model_config_dataset_read_d[-last_n_results:]

%reset -f

last_n_results = 30
results_path = '/usr/local/bin/marketagent/jupyterlab-notebooks/active-working-notebooks/RL/models/dqn-model-config.csv'

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

class FlowLayout(object):
    ''' A class / object to display plots in a horizontal / flow layout below a cell '''
    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml =  """
        <style>
        .floating-box {
        display: inline-block;
        margin: 0px;
        border: 0px solid #888888;  
        }
        </style>
        """
    def add_plot(self, oAxes):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio=io.BytesIO() # bytes buffer for the plot
        fig = oAxes.get_figure()
        fig.canvas.print_png(Bio) # make a png of the plot in the buffer

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml+= (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')
    def PassHtmlToCell(self):
        ''' Final step - display the accumulated HTML '''
        display(HTML(self.sHtml))
        
results_path = '/usr/local/bin/marketagent/jupyterlab-notebooks/active-working-notebooks/RL/models/dqn-model-config.csv'


model_config_dataset_read = pd.read_csv(results_path, index_col=0)

model_config_dataset_read = model_config_dataset_read[-last_n_results:]

fig_size = (6,5)
header_font_size = 20
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)
bar_width = .5

for i in range(0 , len(model_config_dataset_read)) : 
    
    print('Result' , i)
    
    oPlot = FlowLayout() # create an empty FlowLayout
    
    try:
        loss_values = literal_eval(model_config_dataset_read['loss_values'].iloc[i])
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title("Policy Network Loss" , fontsize=header_font_size)
        ax.set_xlabel('Epoch', fontsize = 15.0) 
        ax.set_ylabel('Loss', fontsize = 15.0) 
        ax.grid()
        ax.plot(loss_values)
        oPlot.add_plot(ax)
        plt.close()
    except Exception as e:
        print("An exception occurred in iteration" , i , e) 
        
    try:
        all_reward = literal_eval(model_config_dataset_read['all_reward'].iloc[i])
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title("Cumulative Reward - Per Epoch" , fontsize=header_font_size)
        ax.set_xlabel('Epoch', fontsize = 15.0) 
        ax.set_ylabel('Reward', fontsize = 15.0) 
        ax.grid()
        ax.plot(all_reward)
        oPlot.add_plot(ax)
        plt.close()
    except Exception as e:
        print("An exception occurred in iteration" , i , e) 
        
    try:
        std_predictions_training = literal_eval(model_config_dataset_read['std_predictions_training'].iloc[i])
        std_predictions_test = literal_eval(model_config_dataset_read['std_predictions_test'].iloc[i])
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title("$\sigma$ Predictions", fontsize=header_font_size)
        ax.plot(std_predictions_training , label='Training')
        ax.plot(std_predictions_test , label='Test')
        ax.legend()
        ax.set_xlabel('Episode', fontsize = 15.0) 
        ax.set_ylabel('$\sigma$', fontsize = 15.0) 
        ax.grid()
        oPlot.add_plot(ax)
        plt.close()
    except Exception as e:
        print("An exception occurred in $\sigma$ Predictions, iteration" , i , e) 
       
    try:
        model_actions_training = literal_eval(model_config_dataset_read['model_actions_training'].iloc[i])
        c = Counter(model_actions_training)
        height = list(c.keys())
        x = list(c.values())
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title("Training set actions count", fontsize=header_font_size)
        ax.set_xlabel('Training Instance', fontsize = 15.0) 
        ax.set_ylabel('Predictions Count', fontsize = 15.0) 
        width = .8
        ax.bar(x, height, width, color='cornflowerblue' )
#         plt.xticks(x) 
#         plt.xticks(list(range(1,161))) 
        oPlot.add_plot(ax)
        plt.close()
    except Exception as e:
        print("Predictions Count - Training actions" , i , e) 
        
    try:
        model_actions_test = literal_eval(model_config_dataset_read['model_actions_test'].iloc[i])
        c = Counter(model_actions_test)
        height = list(c.keys())
        x = list(c.values())
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title("Test set actions count", fontsize=header_font_size)
        ax.set_xlabel('Training Instance', fontsize = 15.0) 
        ax.set_ylabel('Predictions Count', fontsize = 15.0) 
        ax.bar(x, height, bar_width, color='cornflowerblue' )
#         plt.xticks(x) 
        oPlot.add_plot(ax)
        plt.close()
    except Exception as e:
        print("Predictions Count - Training actions" , i , e) 

    oPlot.PassHtmlToCell()

    resultDataExecutionTime = np.array([model_config_dataset_read.iloc[i]['execution_time']])
    resultDataPeriodShiftSize = np.array([model_config_dataset_read.iloc[i]['PeriodShiftSize']])
    resultDataDQNInputSize = np.array([model_config_dataset_read.iloc[i]['dqn_input_size_config']])
    resultDataWindowSize = np.array([model_config_dataset_read.iloc[i]['w_size']])
    resultDataLR = np.array([model_config_dataset_read.iloc[i]['learning_rate']])
    resultDataEpsilon = np.array([model_config_dataset_read.iloc[i]['epsilon']])
    resultDataGamma = np.array([model_config_dataset_read.iloc[i]['gamma']])
    resultDataNEpisodes = np.array([model_config_dataset_read.iloc[i]['number_episodes']])
    resultDataNEpochs = np.array([model_config_dataset_read.iloc[i]['number_epochs']])
    resultDataModelExecutionTime = np.array([(model_config_dataset_read.iloc[i]['end_time'] - model_config_dataset_read.iloc[i]['start_time']) / 1000 / 60])
    resultDataPeriodShiftSize = np.array([model_config_dataset_read.iloc[i]['PeriodShiftSize']])
    resultDataOptomizer = np.array([model_config_dataset_read.iloc[i]['PeriodShiftSize']])
    training_set_actions_count = np.array([model_config_dataset_read.iloc[i]['training_set_actions_count']])
    test_set_actions_count = np.array([model_config_dataset_read.iloc[i]['test_set_actions_count']])
    total_episode_reward_sum = np.array([model_config_dataset_read.iloc[i]['total_reward_sum_episodes']])

    resultDataSet = pd.DataFrame()
    resultDataSet['ExecutionTime'] = resultDataExecutionTime.tolist()
    resultDataSet['PeriodShiftSize'] = resultDataPeriodShiftSize.tolist()
    resultDataSet['DQN Input Size'] = resultDataDQNInputSize.tolist()
    resultDataSet['WindowSize'] = resultDataWindowSize.tolist()
    resultDataSet['LR'] = resultDataLR.tolist()
    resultDataSet['Epsilon'] = resultDataEpsilon.tolist()
    resultDataSet['Gamma'] = resultDataGamma.tolist()
    resultDataSet['# Episodes'] = resultDataNEpisodes.tolist()
    resultDataSet['# Epochs'] = resultDataNEpochs.tolist()
    resultDataSet['Model Exec Time'] = resultDataModelExecutionTime.tolist()
    resultDataSet['Optimizer'] = resultDataOptomizer.tolist()
    resultDataSet['Training Set Actions Count'] = training_set_actions_count.tolist()
    resultDataSet['Tes Set Actions Count'] = test_set_actions_count.tolist()
    resultDataSet['Reward all episodes sum'] = total_episode_reward_sum

#     print(resultDataSet) 
    
    print('****************************************************************************************************************************************************************************************************************************************')
    
    total_reward_sum_episodes_adam = list(model_config_dataset_read[model_config_dataset_read['optimizer'] == 'adam']['total_reward_sum_episodes'])
total_reward_sum_episodes_sgd = list(model_config_dataset_read[model_config_dataset_read['optimizer'] == 'SGD']['total_reward_sum_episodes'])

print('total_reward_sum_episodes_adam' , len(total_reward_sum_episodes_adam))
print('total_reward_sum_episodes_sgd' , len(total_reward_sum_episodes_sgd))

N = 15

ind = np.arange(N)    

fig = plt.figure()
width = .5
plt.figure(figsize=(11,7))

p1 = plt.bar(ind, total_reward_sum_episodes_adam[:N], width)
p2 = plt.bar(ind, total_reward_sum_episodes_sgd[:N], width)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.title('Total reward for each experiment', fontsize = 20.0)
plt.xlabel('Experiment', fontsize = 15.0)
plt.ylabel('Total Reward for all episodes', fontsize = 15.0)
plt.legend((p1[0], p2[0]), ('Adam', 'SGD'))

plt.show()
