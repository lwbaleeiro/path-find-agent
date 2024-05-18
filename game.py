import pygame
import numpy as np
import os
import random
import sys

from agent import Agent
from object import ObstaclesPhaseOne, Goal

# Inicializa o Pygame
pygame.init()

# Define as dimensões da janela do jogo
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Agente que Evita Obstáculos')

generations = 1000

# Define as cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Função para obter os inputs da rede neural
def get_inputs(agent):
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

# Função principal do jogo
def main():
    run = True
    clock = pygame.time.Clock()

    obstacles_phase_one = ObstaclesPhaseOne(window)
    goal = Goal()

    for i in range(generations):
        print(f"Generation: {i}")
        agent = Agent()
        run = True

        # Carrega os pesos da rede neural se o arquivo existir
        # if os.path.exists("weights.pkl"):
        #     agent.q_network.load_weights()

        while run:
            clock.tick(30)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Obtém os inputs da rede neural
            inputs = get_inputs(agent)
            action = agent.act(inputs)

            # Move o agente com base na ação selecionada
            agent.move(action)
            
            # Verifica colisões
            reward, run = agent.check_collision(obstacles_phase_one.obstacles_list, goal)

            if not run:
                print("Atingiu obstaculo")
                
            # Verifica se tem passos sobrando
            if run == True and agent.steps == 0:
                print("Acabou os passos")
                run = False

            # # Salva os pesos da rede neural após cada episódio
            if not run and reward == 100:
                print("Atingiu o objetivo")
                agent.q_network.save_weights()
            
            agent.train(inputs, reward)

            window.fill(WHITE)
            agent.draw(window)
            
            obstacles_phase_one.draw()

            goal.draw(window)
            
            pygame.display.update()

    agent.q_network.save_weights()         
    if not run:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
