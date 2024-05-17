import pygame
import random

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
window = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
pygame.display.set_caption("Path find agent")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Agent:
    def __init__(self):
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.size = 20
        self.color = GREEN
        self.speed = 5

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = RED

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.width, self.height))

class Goal:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = BLACK

    def draw(self, window):
        pygame.draw.circle(window, self.color, (self.x, self.y), self.radius)

def main():
    run = True
    clock = pygame.time.Clock()

    agent = Agent()
    obstacles = [Obstacle(random.randint(0, WINDOW_WIDTH-50), random.randint(0, WINDOW_HEIGHT-50), 50, 50) for _ in range(5)]
    goal = Goal(20, 30, 6)
    
    while run:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            agent.move(-agent.speed, 0)
        if keys[pygame.K_RIGHT]:
            agent.move(agent.speed, 0)
        if keys[pygame.K_UP]:
            agent.move(0, -agent.speed)
        if keys[pygame.K_DOWN]:
            agent.move(0, agent.speed)
        
        window.fill(WHITE)
        agent.draw(window)
        for obstacle in obstacles:
            obstacle.draw(window)
        goal.draw(window)
        
        pygame.display.update()
    
    pygame.quit()

if __name__ == "__main__":
    main()