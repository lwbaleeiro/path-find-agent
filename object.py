import pygame

# Classe para o obstáculo
class Obstacle:
    def __init__(self, window, x, y):
        self.x = x
        self.y = y
        self.window = window
        self.width = 50
        self.height = 50
        self.color = (255, 0, 0)
    
    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.width, self.height))    

class ObstaclesPhaseOne:
    def __init__(self, window):
        self.window = window
        self.obstacles_list = self.__get_obstables()

    def __get_obstables(self):
        obstacles = []
        obstacles_coord = [(438,116), (949,508), (29,347), (940,371), (243,78)]
        for coord in obstacles_coord:
            obstacles.append(Obstacle(self.window, coord[0], coord[1])) 

        return obstacles
    
    def draw(self):
        for obstable in self.obstacles_list:
            obstable.draw()

# Classe para o objetivo
class Goal:
    def __init__(self, x = 1150, y = 300, size = 24):
        self.x = x
        self.y = y
        self.size = size
        self.color = (0, 154, 154)
    
    def draw(self, window):
        pygame.draw.circle(window, self.color, (self.x, self.y), self.size)