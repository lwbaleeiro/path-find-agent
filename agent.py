import random
from neural_network import NeuralNetwork
import numpy as np
import pygame

class Agent:
    def __init__(self, input_size=8, hidden_size=16, output_size=2):

        self.surface = pygame.display.get_surface() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.q_network = NeuralNetwork(self.input_size, hidden_size, self.output_size)

        self.size = 20
        self.x = 10
        self.y = 300
        self.color = (0, 255, 0)
        self.speed = 5

        self.steps = 1000

    def move(self, action):
        
        # prev_x = self.x
        # prev_y = self.y

        # Verifica se a saída da rede neural é uma matriz bidimensional
        # Move o agente com base na ação selecionada
        if len(action.shape) == 1:
            dx = (action[0] - 0.5) * 2 * self.speed
            dy = (action[1] - 0.5) * 2 * self.speed
        else:
            dx = (action[0][0] - 0.5) * 2 * self.speed
            dy = (action[0][1] - 0.5) * 2 * self.speed

        # Verifica se o agente vai colidir com as bordas da tela
        if 0 <= self.x + dx <= self.surface.get_width() - self.size:
            self.x += dx
        if 0 <= self.y + dy <= self.surface.get_height() - self.size:
            self.y += dy

        # # Verifica se ele saiu do lugar
        # if (self.x != prev_x) or (self.y != prev_y):
        self.steps -= 1
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))
    
    # Função para verificar colisões
    def check_collision(self, obstacles, goal):
        for obstacle in obstacles:
            if (self.x < obstacle.x + obstacle.width and
                self.x + self.size > obstacle.x and
                self.y < obstacle.y + obstacle.height and
                self.y + self.size > obstacle.y):
                return -200, False
        
        if (self.x < goal.x + goal.size and
            self.x + self.size > goal.x and
            self.y < goal.y + goal.size and
            self.y + self.size > goal.y):
            return 100, False
        
        return 0, True
        
    # Seleciona uma ação com base na política ε-greedy
    def act(self, inputs):
        if random.uniform(0, 1) <= self.q_network.epsilon:
            return np.random.rand(2)
        return self.q_network.forward(inputs) * 2 - 1
    
    def train(self, next_inputs, reward):  
        return self.q_network.backward(next_inputs, reward)