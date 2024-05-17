import pygame
import numpy as np
import pickle
import os
import random

# Inicializa o Pygame
pygame.init()

# Define as dimensões da janela do jogo
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Agente que Evita Obstáculos')

# Define as cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Classe para o agente
class Agent:
    def __init__(self):
        self.size = 20
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.color = GREEN
        self.speed = 5
    
    def move(self, dx, dy):
        # Verifica se o agente vai colidir com as bordas da tela
        if 0 <= self.x + dx <= WINDOW_WIDTH - self.size:
            self.x += dx
        if 0 <= self.y + dy <= WINDOW_HEIGHT - self.size:
            self.y += dy
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))

# Classe para o obstáculo
class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = RED
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.width, self.height))

# Classe para o objetivo
class Goal:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.color = BLACK
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))

# Classe para a rede neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializa os pesos
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Inicializa os bias
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, inputs, target, learning_rate):
        output_error = target - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        # Atualiza os pesos e bias
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.reshape(-1, 1).dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

# Função para verificar colisões
def check_collision(agent, obstacles, goal):
    for obstacle in obstacles:
        if (agent.x < obstacle.x + obstacle.width and
            agent.x + agent.size > obstacle.x and
            agent.y < obstacle.y + obstacle.height and
            agent.y + agent.size > obstacle.y):
            return True
    
    if (agent.x < goal.x + goal.size and
        agent.x + agent.size > goal.x and
        agent.y < goal.y + goal.size and
        agent.y + agent.size > goal.y):
        return True
    
    return False

# Função para obter os inputs da rede neural
def get_inputs(agent, obstacles, goal):
    inputs = np.zeros(8)  # 8 inputs: distâncias nas direções N, NE, E, SE, S, SW, W, NW
    inputs[0] = agent.y  # Norte
    inputs[1] = min(WINDOW_WIDTH - agent.x, WINDOW_HEIGHT - agent.y)  # Nordeste
    inputs[2] = WINDOW_WIDTH - agent.x  # Leste
    inputs[3] = min(WINDOW_WIDTH - agent.x, agent.y)  # Sudeste
    inputs[4] = WINDOW_HEIGHT - agent.y  # Sul
    inputs[5] = min(agent.x, WINDOW_HEIGHT - agent.y)  # Sudoeste
    inputs[6] = agent.x  # Oeste
    inputs[7] = min(agent.x, agent.y)  # Noroeste

    # Normaliza os inputs
    inputs /= max(WINDOW_WIDTH, WINDOW_HEIGHT)

    return inputs

# Função para salvar os pesos da rede neural em um arquivo
def save_weights(neural_network, filename):
    with open(filename, 'wb') as f:
        pickle.dump(neural_network.weights_input_hidden, f)
        pickle.dump(neural_network.weights_hidden_output, f)

# Função para carregar os pesos da rede neural de um arquivo
def load_weights(neural_network, filename):
    with open(filename, 'rb') as f:
        neural_network.weights_input_hidden = pickle.load(f)
        neural_network.weights_hidden_output = pickle.load(f)

# Função principal do jogo
def main():
    run = True
    clock = pygame.time.Clock()
    
    agent = Agent()
    obstacles = [Obstacle(random.randint(0, WINDOW_WIDTH-50), random.randint(0, WINDOW_HEIGHT-50), 50, 50) for _ in range(5)]
    goal = Goal(WINDOW_WIDTH-40, WINDOW_HEIGHT-40, 40)
    
    neural_network = NeuralNetwork(input_size=8, hidden_size=16, output_size=2)
    learning_rate = 0.01
    gamma = 0.9  # Fator de desconto para recompensas futuras
    epsilon = 1.0  # Taxa de exploração inicial
    epsilon_decay = 0.995  # Taxa de decaimento da taxa de exploração
    min_epsilon = 0.01  # Taxa de exploração mínima
    total_reward = 0
    
    # Carrega os pesos da rede neural se o arquivo existir
    if os.path.exists("weights.pkl"):
        load_weights(neural_network, "weights.pkl")

    while run:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        # Obtém os inputs da rede neural
        inputs = get_inputs(agent, obstacles, goal)
        
        # Seleciona uma ação com base na política ε-greedy
        if random.uniform(0, 1) < epsilon:
            action = np.random.rand(2)
        else:
            action = neural_network.forward(inputs) * 2 - 1  # Normaliza a saída para o intervalo [-1, 1]
        #print("Action:", action)

        # Move o agente com base na ação selecionada
        # Verifica se a saída da rede neural é uma matriz bidimensional
        # Move o agente com base na ação selecionada
        if len(action.shape) == 1:
            dx = (action[0] - 0.5) * 2 * agent.speed
            dy = (action[1] - 0.5) * 2 * agent.speed
        else:
            dx = (action[0][0] - 0.5) * 2 * agent.speed
            dy = (action[0][1] - 0.5) * 2 * agent.speed

        agent.move(dx, dy)
        
        # Verifica colisões
        if check_collision(agent, obstacles, goal):
            reward = -100  # Penalidade por colidir com obstáculos
            run = False
        elif agent.x > goal.x and agent.y > goal.y:
            reward = 100  # Recompensa por alcançar o objetivo
            run = False
        else:
            reward = 0  # Recompensa neutra se o agente estiver em movimento

        # Salva os pesos da rede neural após cada episódio
        if not run:
            save_weights(neural_network, "weights.pkl")
        
        total_reward += reward
        
        # Calcula o próximo estado
        next_inputs = get_inputs(agent, obstacles, goal)
        
        # Atualiza a Q-Value com o algoritmo Q-Learning
        target = reward + gamma * np.max(neural_network.forward(next_inputs))
        neural_network.backward(inputs, target, learning_rate)
        
        # Atualiza a taxa de exploração (epsilon)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        window.fill(WHITE)
        agent.draw(window)
        for obstacle in obstacles:
            obstacle.draw(window)
        goal.draw(window)
        
        pygame.display.update()
    

    print("Total Reward:", total_reward)
    pygame.quit()

if __name__ == "__main__":
    main()
