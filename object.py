import pygame

# Classe para o obstáculo
class Obstacle:
    def __init__(self, window, x, y, width, height):
        self.x = x
        self.y = y
        self.window = window
        self.width = width
        self.height = height
        self.color = (255, 0, 0)
    
    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.width, self.height))    

class ObstaclesPhaseOne:
    def __init__(self, window):
        self.window = window
        self.width = 50
        self.height = 70
        self.obstacles_list = self.__get_obstables()

    def __get_obstables(self):
        obstacles = []
        obstacles_coord = [(250, 250), (400, 400), (550, 150), (700, 100), (700, 500)]
        for coord in obstacles_coord:
            obstacles.append(Obstacle(self.window, coord[0], coord[1], self.width, self.height)) 

        return obstacles
    
    def draw(self):
        for obstable in self.obstacles_list:
            obstable.draw()

# Classe para o objetivo
class Goal:
    def __init__(self, size = 24):
        self.x = pygame.display.get_surface().get_width() - 50
        self.y = pygame.display.get_surface().get_height() // 2
        self.size = size
        self.color = (0, 154, 154)
    
    def draw(self, window):
        pygame.draw.circle(window, self.color, (self.x, self.y), self.size)