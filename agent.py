import random
from neural_network import NeuralNetwork
import numpy as np
import pygame
from collections import deque
from replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, input_size=8, hidden_size=16, output_size=2, replay_buffer = None):

        self.surface = pygame.display.get_surface() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.q_network = NeuralNetwork(self.input_size, hidden_size, self.output_size)
        self.replay_buffer = replay_buffer

        self.size = 20
        self.x = 100
        self.y = 300
        self.color = (0, 255, 0)
        self.speed = 5
        self.alive = True

        self.collided = False
        self.reached_goal = False
        self.reward = 0
        self.next_action = np.random.rand(2)
        self.inputs = np.zeros(8)  # 8 inputs: distâncias nas direções N, NE, E, SE, S, SW, W, NW
        self.memory = deque(maxlen=2000)

        self.steps = 6000

    def move(self):
        if self.alive == True:
            action = self.next_action 
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

            self.steps -= 1
            self.reward -= 0.01
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))
    
    # Função para verificar colisões
    def check_collision(self, obstacles, goal):
        self.reward = - 0.1
        for obstacle in obstacles:
            if (self.x < obstacle.x + obstacle.width and
                self.x + self.size > obstacle.x and
                self.y < obstacle.y + obstacle.height and
                self.y + self.size > obstacle.y):
                self.collided = True
                self.reward = -100

        # Verifica se o agente vai colidir com as bordas da tela
        if (self.x >= self.surface.get_width() - self.size or self.x <= 0 or 
            self.y >= self.surface.get_height() - self.size or self.y <= 0):
            self.collided = True
            self.reward = -100

        if (self.x < goal.x + goal.size and
            self.x + self.size > goal.x and
            self.y < goal.y + goal.size and
            self.y + self.size > goal.y):
            self.reached_goal = True
            self.reward = 100
        else:
            # Recompensa baseada na proximidade ao objetivo
            distance_to_goal = np.sqrt((self.x - goal.x) ** 2 + (self.y - goal.y) ** 2)
            self.reward += 10.0 / distance_to_goal
        
    # Seleciona uma ação com base na política ε-greedy
    def act(self):
        self.__get_inputs()

        if random.uniform(0, 1) <= self.q_network.epsilon:
            self.next_action = np.random.rand(2)
        else:
            self.next_action = self.q_network.forward(self.inputs)

    # Armazenar as experiências passadas e usar mini-lotes para treinamento pode melhorar a estabilidade e a eficiência do treinamento.
    def store_experience(self, next_state, done):
        self.replay_buffer.add(self.inputs, self.next_action, self.reward, next_state, done)

    def train_from_replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.q_network.gamma * np.max(self.q_network.forward(next_state))
            
            self.q_network.backward(state, target)           

    # Função para obter os inputs da rede neural
    def __get_inputs(self):
        
        self.inputs[7] = self.y  # Norte
        self.inputs[6] = min(self.surface.get_width() - self.x, self.surface.get_width() - self.y)  # Nordeste
        self.inputs[5] = self.surface.get_width() - self.x  # Leste
        self.inputs[4] = min(self.surface.get_width() - self.x, self.y)  # Sudeste
        self.inputs[3] = self.surface.get_height() - self.y  # Sul
        self.inputs[2] = min(self.x, self.surface.get_height() - self.y)  # Sudoeste
        self.inputs[1] = self.x  # Oeste
        self.inputs[0] = min(self.x, self.y)  # Noroeste

        # Normaliza os inputs
        self.inputs /= max(self.surface.get_width(), self.surface.get_height())

############################################################################################
################################## CLASS POPULATION AGENT ##################################
############################################################################################

class PopulationAgent():
    def __init__(self, population = 50, replay_buffer_size = 10000):
        self.population_number = population
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.population_agent_list = self.__get_population()
        self.agents_reached_goal = []

    def __get_population(self):
        agent_list = []
        for _ in range(self.population_number):
            agent_list.append(Agent(replay_buffer=self.replay_buffer))
        return agent_list

    def move(self):
        for agent in self.population_agent_list:
            # Verifica se tem passos sobrando
            if agent.steps == 0:
                agent.alive = False
                self.population_agent_list.remove(agent)
            else:
                agent.move()
    
    def draw(self, window):
        for agent in self.population_agent_list:
            agent.draw(window)
    
    def check_collision(self, obstacles, goal):
        for agent in self.population_agent_list:
            agent.check_collision(obstacles, goal)
            if agent.collided:
                agent.alive = False
                self.population_agent_list.remove(agent)
            elif agent.reached_goal:
                self.agents_reached_goal.append(agent)
                self.population_agent_list.remove(agent)
    
    def check_all_dead_or_rechad_goal(self):
        for agent in self.population_agent_list:
            if agent.alive or agent.reached_goal:
                return False  
                 
        return True
    
    def save_best_agent(self):
        less_steps_take = 1000
        best_agent = None

        for agent in self.agents_reached_goal:
            if agent.steps < less_steps_take:
                less_steps_take = agent.steps
                best_agent = agent

        if best_agent:
            best_agent.q_network.save_weights("")

    def load_weights(self):
        self.population_agent_list[0].q_network.load_weights("")

    def act(self):
        for agent in self.population_agent_list:
            agent.act()
    
    def train(self):
        for agent in self.population_agent_list:
            if agent.alive or agent.reached_goal:
                next_state = agent.inputs.copy() # Copia dos inputs recebido
                agent.store_experience(next_state, agent.reached_goal)
                agent.train_from_replay(batch_size = 32)