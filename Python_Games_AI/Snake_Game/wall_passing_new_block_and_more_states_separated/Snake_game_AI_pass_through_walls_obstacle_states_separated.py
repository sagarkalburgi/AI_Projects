# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:10:36 2021

@author: LENOVO
"""

import pygame
import random
import time
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()

font = pygame.font.SysFont('airal', 25)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    #NONE = 5
    

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 40
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
TURNS = 0
RED = (200, 0, 0)

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.h = h
        self.w = w
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
       
        #self._place_block()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.block = None
        self._place_food()
        self._place_block()
        self.frame_iteration = 0
        
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        
        if self.food in self.snake:
            self._place_food()
            
    def _place_block(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.block = Point(x, y)
        if (self.food.x == self.block.x and self.food.y == self.block.y) or (self.block in self.snake):
            self._place_block()
     
    def play_step(self, action):
        self.frame_iteration += 1
        
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                    
        # move
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        
        # check if game over
        game_over = False
        self.wall_passing()
        if self.is_collision() or self.frame_iteration > 100*len(self.snake) or self.block == self.head:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # place new food and block or just move
        if self.head == self.food:
            self.score += 1
            reward = +10
            self._place_food()
            self._place_block()
        
        else:
            self.snake.pop()
            
        if self.head == self.block:
            game_over = True
            reward = -50
            return reward, game_over, self.score
            
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        # hits boundary
        # if self.direction == Direction.NONE:
        #     return False
        if pt is None:
            pt = self.head
        
        # hits itself
        if len(pt) == 2 and pt in self.snake[1:]:
            # text = font_init.render("Game Over", True, WHITE)
            # self.display.blit(text, [self.w/3.5,self.h/3.5])
            # pygame.display.flip()
            # time.sleep(3)
            return True
        
        # for i in pt:
        #     if i in self.snake[1:]:
        #         return True
        
        return False
    
    def wall_passing(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE:
            self.head = Point(0 - BLOCK_SIZE, self.head.y)
            
        if pt.x < 0:
            self.head = Point(self.w, self.head.y)
            
        if pt.y > self.h - BLOCK_SIZE:
            self.head = Point(self.head.x, 0 - BLOCK_SIZE)
            
        if pt.y < 0:
            self.head = Point(self.head.x, self.h)
            
        return
        
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.block.x, self.block.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        
        # if self.direction == Direction.NONE:
        #     text = font_init.render("Start Game", True, WHITE)
        #     self.display.blit(text, [self.w/3.5,self.h/3.5])
        #     text = font1.render("Eat the green fruit to score", True, WHITE)
        #     self.display.blit(text, [self.w/3,self.h/2.5])
        #     text = font1.render("Avoid hitting the red block", True, WHITE)
        #     self.display.blit(text, [self.w/3,self.h/2.25])
        pygame.display.flip()
        
    def _move(self, action):
        # straight, right, left
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0 ,0]):
            new_dir = clock_wise[idx]   #  no change
        elif np.array_equal(action, [0, 1 ,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn
            
        self.direction = new_dir
            
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
