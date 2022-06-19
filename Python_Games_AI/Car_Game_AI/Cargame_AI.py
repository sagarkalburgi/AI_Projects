# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:00:47 2021

@author: LENOVO
"""

import pygame
import numpy as np
import random
import time
from enum import Enum

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    NONE = 3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

font = pygame.font.SysFont('airal', 25)
font_init = pygame.font.SysFont('airal', 75)
font1 = pygame.font.SysFont('airal', 25)

car_main = pygame.image.load('car1.png')
car_obs_1 = pygame.image.load('car2.png')
car_obs_2 = pygame.image.load('car3.png')
car_obs_3 = pygame.image.load('car4.png')

road = pygame.image.load('road.png')
car_height = 100
car_width = 50
offset = 2
step_size = 10
car_speed = 25

class CarGameAI:
    
    def __init__(self, w=640, h=640):
        self.h = h
        self.w = w
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.car_main_rec = car_main.get_rect()
        self.car_obs_rec_1 = car_obs_1.get_rect()
        self.car_obs_rec_2 = car_obs_2.get_rect()
        self.car_obs_rec_3 = car_obs_3.get_rect()
        
        self.sensor_1 = pygame.Rect(217, 0, 50, 640)
        self.sensor_2 = pygame.Rect(297, 0, 50, 640)
        self.sensor_3 = pygame.Rect(372, 0, 50, 640)
        
        self.car_main_rec.x = self.w/2 - car_width/2 + offset
        self.car_main_rec.y = self.h - car_height

        self.car_obs_rec_1.x, self.car_obs_rec_2.x, self.car_obs_rec_3.x = 217, 297, 372
        self.car_obs_rec_1.y, self.car_obs_rec_2.y, self.car_obs_rec_3.y = -100, -100, -100
        
        self.car_gamer = [self.car_main_rec.x, self.car_main_rec.y]
        self.car_obs_pos = [[self.car_obs_rec_1.x, self.car_obs_rec_1.y], [self.car_obs_rec_2.x, self.car_obs_rec_2.y], [self.car_obs_rec_3.x, self.car_obs_rec_3.y]]

        self.direction = Direction.RIGHT
        self.score = 0
        self.state = 1
        self.state_word = ['Left', 'Middle', 'Right']
        self.car_obs = [car_obs_1, car_obs_2, car_obs_3]
        self.x_pos = [217, 297, 372]
        self.game_started = True
        self._reset_states()
        
    def play_step(self, action):
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action)
        self.car_position()
        
        # check if game over
        game_over = False
        reward = 0
        if self._is_collision(self.car_obs_rec_1) or self._is_collision(self.car_obs_rec_2) or self._is_collision(self.car_obs_rec_3):
            game_over = True
            reward = -10
            return reward, game_over, self.score 
        
        # Add new cars
        for i in range(0, 3):
            if self.car_obs_pos[i][1] > self.h:
                self.score += 1
                reward = +10
                self.car_obs = random.sample(self.car_obs, len(self.car_obs))
                self._reset_states()
        
            if self.score == 500:
                game_over = True
                return reward, game_over, self.score 
        # update ui and clock
        self._update_ui()
        self.clock.tick(min(self.score + step_size*5, 60))
        
        # return game over and score
        return reward, game_over, self.score
    
    def car_position(self):
        if not self.game_started:                
            for i in self.car_list:
                self.car_obs_pos[i] = [self.x_pos[i], -100]
                        
        if self.game_started:
            for i in self.car_list:
                self._move_car(i, self.car_obs_pos[i])

    def _car_load(self, car, cord):
        self.display.blit(car, (cord[0], cord[1]))
    
    def _update_ui(self):
        self.display.fill(WHITE)
        self.display.blit(road, (0,0))
        
        # if not self.game_started:
        #     text = font_init.render("Start Game", True, WHITE)
        #     self.display.blit(text, [self.w/3.5,self.h/3.5])
        #     text = font1.render("Avoid hitting other cars", True, WHITE)
        #     self.display.blit(text, [self.w/3,self.h/2.5])
        
        text = font1.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [10,10])
        text = font1.render("Lane: " + self.state_word[self.state], True, BLACK)
        self.display.blit(text, [10,30])
        self._car_load(car_main, self.car_gamer)
        self._car_load(self.car_obs[0], self.car_obs_pos[0])
        self._car_load(self.car_obs[1], self.car_obs_pos[1])
        self._car_load(self.car_obs[2], self.car_obs_pos[2])
        pygame.display.flip()
        
    def _move(self, action):
        
        if np.array_equal(action, [1, 0 ,0]):
            new_state = 0
        elif np.array_equal(action, [0, 1 ,0]):
            new_state = 1 
        else:
            new_state = 2
        
        x = self.car_gamer[0]
        y = self.car_gamer[1]
        if new_state == 0:
            x = 219
            self.car_main_rec.x = 217
            self.state = 0
        if new_state == 1:
            x = 297
            self.car_main_rec.x = 297
            self.state = 1
        if new_state == 2:
            x = 372
            self.car_main_rec.x = 372
            self.state = 2
        
        self.car_gamer = [x, y]
        
    def _move_car(self, num, position):
        y = position[1]
        y += car_speed
        self.car_obs_pos[num] = [position[0], y]
        if num == 0:
            self.car_obs_rec_1.y += car_speed
        
        if num == 1:
            self.car_obs_rec_2.y += car_speed
            
        if num == 2:
            self.car_obs_rec_3.y += car_speed
        
    def _reset_states(self):
        for i in range(0, 3):
            self.car_obs_pos[i] = [self.x_pos[i], -100]
        
        self.car_obs_rec_1.x, self.car_obs_rec_2.x, self.car_obs_rec_3.x = 217, 297, 372
        self.car_obs_rec_1.y, self.car_obs_rec_2.y, self.car_obs_rec_3.y = -100, -100, -100
        
        self.car_gamer = [self.car_main_rec.x, self.car_main_rec.y]
        self.car_obs_pos = [[self.car_obs_rec_1.x, self.car_obs_rec_1.y], [self.car_obs_rec_2.x, self.car_obs_rec_2.y], [self.car_obs_rec_3.x, self.car_obs_rec_3.y]]
        
        self.car_list = []
        for i in range(0, np.random.randint(2) + 1):
            self.car_list.append(np.random.randint(3))
            
    def _is_collision(self, car_obs_rec):
        if self.car_main_rec.colliderect(car_obs_rec):
            return True
            
        return False
    
    def _get_sensor_response(self, sensor):
        if self.car_obs_rec_1.colliderect(sensor) or self.car_obs_rec_2.colliderect(sensor) or self.car_obs_rec_3.colliderect(sensor):
            return True
        
        return False
