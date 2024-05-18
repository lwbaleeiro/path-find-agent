import pygame

# Classe para o obstáculo
class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 0, 0)
    
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.width, self.height))

# Classe para o objetivo
class Goal:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.color = (0, 154, 154)
    
    def draw(self, window):
        pygame.draw.circle(window, self.color, (self.x, self.y), self.size)