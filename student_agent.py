# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict
from Game2048Env import Game2048Env
from Approximator import NTupleApproximator
from UCTMCTS import *
import gdown

# 方法 1：使用模糊匹配（自動轉換連結）
url = 'https://drive.google.com/file/d/1o4NDbGB03qLXtYKSDBBul6MqYUq37n0x/view?usp=share_link'
gdown.download(url, output='downloaded_file.pkl', quiet=False, fuzzy=True)

def get_action(state, score):
    env = Game2048Env()
    approximator = NTupleApproximator.load_model("downloaded_file.pkl")
    
    def value_function(state):
        return approximator.value(state)
    iterations = 500
    exploration_constant = 0.01
    
    # Create the MCTS agent
    mcts = UCTMCTS(env, value_function, iterations, exploration_constant)
    
    # Play the game
    state = env.reset()
    done = False
    env.state = state
    env.score = score
        # Run MCTS to get the best action
    best_action, action_distribution = mcts.run_mcts(state, env.score)
        
        # Take the best action
    return best_action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


