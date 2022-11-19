from turtle import circle
import pandas as pd
from pyglet import shapes, window

#windows lines hardcoded
window_lines = [
    [(0, 0), (640, 0)],
    [(640, 0), (640, 480)],
    [(0, 0), (0, 480)],
    [(0, 480), (640, 480)]
]

reward_gates = []
big_prize = 0
window_dimensions = (0, 0)


#obstacle
obstacles = []
obstaclesDf = []
class rect:
    def __init__(self, x, y, width, height, new_color,rotation, batch):
        self.sprite = (shapes.Rectangle(x, y, height, width, color=new_color, batch=batch) if rotation == -90
                    else shapes.Rectangle(x, y, width, height, color=new_color, batch=batch))
        self.rotation = rotation
        self.obstacle_lines = [
            [(self.sprite.x, self.sprite.y), (self.sprite.x + self.sprite.width, self.sprite.y)],
            [(self.sprite.x + self.sprite.width, self.sprite.y), (self.sprite.x + self.sprite.width, self.sprite.y + self.sprite.height)],
            [(self.sprite.x + self.sprite.width, self.sprite.y + self.sprite.height), (self.sprite.x, self.sprite.y + self.sprite.height)],
            [(self.sprite.x, self.sprite.y + self.sprite.height), (self.sprite.x, self.sprite.y)]
        ]
        self.df = pd.DataFrame(data={
            "x1": [self.sprite.x, self.sprite.x + self.sprite.width, self.sprite.x + self.sprite.width, self.sprite.x],
            "y1": [self.sprite.y, self.sprite.y, self.sprite.y + self.sprite.height, self.sprite.y + self.sprite.height],
            "x2": [self.sprite.x + self.sprite.width, self.sprite.x + self.sprite.width, self.sprite.x, self.sprite.x],
            "y2": [self.sprite.y, self.sprite.y + self.sprite.height, self.sprite.y + self.sprite.height, self.sprite.y]
        })

def initObstacle(batch):
    global obstacles
    global obstaclesDf
    obstacles = [
        rect(0, 150, 350, 20, (255, 20, 0),0, batch),
        rect(250, 350, 400, 20,(255, 20, 0),0, batch),
    ]
    obstaclesDf = pd.concat([obstacle.df for obstacle in obstacles])

def init_reward_gates(batch):
    global reward_gates
    global big_prize
    reward_gates = [
        rect(280, 0, 150, 10,(255, 255, 0), -90, batch),
        rect(350, 150+7.5, 640-350, 10,(255, 255, 0), 0, batch),
        
        rect(280, 150+20, 180, 10,(255, 255, 0), -90, batch),
        rect(0, 350+7.5, 640-390, 10,(255, 255, 0), 0, batch),

        rect(280, 350+ 20, 180, 10,(255, 255, 0), -90, batch),
    ]
    for reward_gate in reward_gates:
        reward_gate.sprite.visible = False
    big_prize = shapes.Circle(450, 420, 20, color=(255, 255, 0), batch=batch)

