import random
from neural_network import NeuralNetwork
import numpy as np
import pygame

class Agent:
    def __init__(self, input_size=8, hidden_size=16, output_size=2):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.q_network = NeuralNetwork(self.input_size, hidden_size, self.output_size)

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
    
    def move(self, action):

        # Verifica se a saída da rede neural é uma matriz bidimensional
        # Move o agente com base na ação selecionada
        if len(action.shape) == 1:
            dx = (action[0] - 0.5) * 2 * self.speed
            dy = (action[1] - 0.5) * 2 * self.speed
        else:
            dx = (action[0][0] - 0.5) * 2 * self.speed
            dy = (action[0][1] - 0.5) * 2 * self.speed

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
    
    # Função para verificar colisões
    def check_collision(self, obstacles, goal):
        for obstacle in obstacles:
            if (self.x < obstacle.x + obstacle.width and
                self.x + self.size > obstacle.x and
                self.y < obstacle.y + obstacle.height and
                self.y + self.size > obstacle.y):
                return -100, False
        
        if (self.x < goal.x + goal.size and
            self.x + self.size > goal.x and
            self.y < goal.y + goal.size and
            self.y + self.size > goal.y):
            return 100, False
        
        return 0, True
        
    # Seleciona uma ação com base na política ε-greedy
    def act(self, inputs):
        if np.random.rand() <= self.q_network.epsilon:
            return random.randrange(self.output_size) 
        act_values = self.q_network.forward(inputs) * 2 - 1
        return np.argmax(act_values[0])
    
    def train(self, inputs, reward, output):
        
        self.q_network.backward(inputs, reward, output)
        if self.q_network.epsilon > self.q_network.epsilon_min:
            self.q_network.epsilon *= self.q_network.epsilon_decay

    # def train(self, state, action, reward, next_state, done):
    #     target = reward
    #     if not done:
    #         target = reward + self.gamma * np.amax(self.q_network.forward(next_state)[0])
        
    #     target_f = self.q_network.forward(state)
    #     target_f[0][action] = target
        
    #     self.q_network.backward(state, target_f, self.learning_rate)
        
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay