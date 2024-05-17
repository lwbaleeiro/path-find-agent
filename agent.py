import random
from neural_network import NeuralNetwork
import numpy as np
import pygame

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = NeuralNetwork(state_size, 24, action_size)
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.size = 20
        self.x = 400
        self.y = 300
        self.color = (0, 255, 0)
        self.speed = 5
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy

        # Impede o agente de sair da tela
        if self.x < 0:
            self.x = 0
        elif self.x + self.size > 800:
            self.x = 800 - self.size
        
        if self.y < 0:
            self.y = 0
        elif self.y + self.size > 600:
            self.y = 600 - self.size
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))
    
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.q_network.forward(state)
        return np.argmax(act_values[0])
    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.q_network.forward(next_state)[0])
        
        target_f = self.q_network.forward(state)
        target_f[0][action] = target
        
        self.q_network.backward(state, target_f, self.learning_rate)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay