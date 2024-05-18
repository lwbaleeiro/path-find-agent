import pygame
import sys

from agent import PopulationAgent
from object import ObstaclesPhaseOne, Goal

pygame.init()

# Define as dimensões da janela do jogo
WINDOW_WIDTH, WINDOW_HEIGHT = 900, 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Pathfinder')
font = pygame.font.Font(None, 26)
background_image = pygame.image.load("background.jpg").convert()
background_image = pygame.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))

generations = 1000

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def display_info(generation, agents):
    text = font.render(f"Generation: {generation}", True, BLACK)
    window.blit(text, (300, 10))

    text = font.render(f"Agents alive: {len(agents.population_agent_list)}", True, BLACK)
    window.blit(text, (500, 10))

# Função principal do jogo
def main():
    run = True
    clock = pygame.time.Clock()

    obstacles = ObstaclesPhaseOne(window)
    goal = Goal()

    agents = PopulationAgent()
    # Carrega os pesos da rede neural se o arquivo existir
    agents.load_weights()

    for gen in range(generations):

        agents = PopulationAgent()
        run = True

        while run:
            clock.tick(30)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            agents.act()
            agents.move()
            agents.check_collision(obstacles.obstacles_list, goal)
            agents.train()

            if agents.check_all_dead_or_rechad_goal() == True:
                agents.save_best_agent()
                run = False

            window.blit(background_image, (0, 0))
            agents.draw(window)
            obstacles.draw()
            goal.draw(window)
            display_info(gen, agents)

            pygame.display.update()   

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
