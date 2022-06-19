# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:41:18 2021

@author: LENOVO
"""

import torch
import random
import numpy as np
from collections import deque
from Snake_game_AI_pass_through_walls_obstacle_states_separated import SnakeGameAI, Direction, Point
from model_pass_through_walls_obstacle_states_separated import Linear_QNet, QTrainer
from helper_pass_through_walls_obstacle_states_separated import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # control randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # pop left
        self.model = Linear_QNet(41, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # model, traininer
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        point_list_l = []
        point_list_r = []
        point_list_u = []
        point_list_d = []
        for i in range(0, 11):
            point_list_l.append(Point(head.x - 20*i, head.y))
            point_list_r.append(Point(head.x + 20*i, head.y))
            point_list_u.append(Point(head.x, head.y - 20*i))
            point_list_d.append(Point(head.x, head.y + 20*i))
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = []
        for i in range(0, 11):
            state.append(# Danger straight
            (dir_r and game.is_collision(point_list_r[i])) or
            (dir_l and game.is_collision(point_list_l[i])) or
            (dir_u and game.is_collision(point_list_u[i])) or
            (dir_d and game.is_collision(point_list_d[i])))
          
        for i in range(0, 11):
            state.append(
            # Danger right
            (dir_u and game.is_collision(point_list_r[i])) or
            (dir_d and game.is_collision(point_list_l[i])) or
            (dir_l and game.is_collision(point_list_u[i])) or
            (dir_r and game.is_collision(point_list_d[i])))
            
        for i in range(0, 11):
            state.append(    
            # Danger left
            (dir_d and game.is_collision(point_list_r[i])) or
            (dir_u and game.is_collision(point_list_l[i])) or
            (dir_r and game.is_collision(point_list_u[i])) or
            (dir_l and game.is_collision(point_list_d[i])))
        
        states = [
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            
            # # Block location
            # game.block.x < game.head.x,  # block left
            # game.block.x > game.head.x,  # block right
            # game.block.y < game.head.y,  # block up
            # game.block.y > game.head.y,  # block down
            
            ]
        
        for i in states:
            state.append(i)
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)    # list of tuples
        
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)    
        self.trainer.train_step(states, actions, rewards, next_states, dones)
            
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while agent.n_games <= 500:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)
            
            # plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()