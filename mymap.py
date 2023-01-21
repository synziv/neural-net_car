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
general_obstacles = []

#STAGE DATA
starting_position = [
    {"x": 100, "y": 100, "rotation": 0},
    {"x": 100, "y": 100, "rotation": 90},
    {"x": 500, "y": 75, "rotation": 180},
]
current_stage = 0

#reward gates
general_reward_gates = []
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
    global general_obstacles
    obstacles = [
        [
            rect(0, 150, 350, 20, (255, 20, 0),0, batch),
            rect(250, 350, 400, 20,(255, 20, 0),0, batch),
        ],
        [
            rect(190, 0, 350, 20, (255, 20, 0),-90, batch),
            rect(410, 120, 400, 20,(255, 20, 0),-90, batch),
        ],
        [
            rect(250, 150, 400, 20, (255, 20, 0),0, batch),
            rect(0, 315, 400, 20,(255, 20, 0),0, batch),
        ],
    ]
    general_obstacles = []
    #obstacles lines
    for stage in range(len(obstacles)):
        general_obstacles.append({
            "x": [],
            "y": [],
            "x&width": [],
            "y&height": []
        })
        obstaclesDf.append([])
        obstaclesDf[stage] = pd.concat([obstacle.df for obstacle in obstacles[stage]])
        for obstacle in obstacles[stage]:
            general_obstacles[stage]["x"].append(obstacle.sprite.x)
            general_obstacles[stage]["y"].append(obstacle.sprite.y)
            general_obstacles[stage]["x&width"].append(obstacle.sprite.x + obstacle.sprite.width)
            general_obstacles[stage]["y&height"].append(obstacle.sprite.y + obstacle.sprite.height)
            for line in window_lines:
                obstaclesDf[stage] = pd.concat([obstaclesDf[stage], pd.DataFrame(data={
                    "x1": [line[0][0]],
                    "y1": [line[0][1]],
                    "x2": [line[1][0]],
                    "y2": [line[1][1]]
                })], ignore_index=True,axis=0) 
    

        
def getStartingPosition():
    global starting_position
    return starting_position[current_stage]

def init_reward_gates(batch):
    global reward_gates
    global big_prize
    global general_reward_gates

    general_reward_gates = []
    reward_gates = [
        [
            rect(280, 0, 150, 10,(255, 255, 0), -90, batch),
            rect(350, 150+7.5, 640-350, 10,(255, 255, 0), 0, batch),
            
            rect(280, 150+20, 180, 10,(255, 255, 0), -90, batch),
            rect(0, 350+7.5, 640-390, 10,(255, 255, 0), 0, batch),

            rect(260, 350+ 20, 180, 10,(255, 255, 0), -90, batch),

            rect(450, 350+ 20, 180, 10,(255, 255, 0), -90, batch),
        ],
        [
            rect(0, 350 - 10, 190, 10,(255, 255, 0), 0, batch),
            rect(210, 350 - 10, 200, 10,(255, 255, 0), 0, batch),
            rect(210, 120, 200, 10,(255, 255, 0), 0, batch),
            rect(430, 120, 220, 10,(255, 255, 0), 0, batch),
            rect(430, 300, 220, 10,(255, 255, 0), 0, batch),
        ],
        [
            
            rect(280, 0, 150, 10, (255, 255, 0), -90, batch),
            rect(0, 150, 640-390, 10, (255, 255, 0), 0, batch),
            rect(280, 150, 180, 10, (255, 255, 0), -90, batch),
            rect(400, 315, 640-370, 10, (255, 255, 0), 0, batch),
            rect(390, 315 + 20, 180, 10, (255, 255, 0), -90, batch),
            rect(180, 315 + 20, 180, 10, (255, 255, 0), -90, batch),
        ],

    ]
    for stage in range(len(obstacles)):
        general_reward_gates.append({
            "x": [],
            "y": [],
            "x&width": [],
            "y&height": []
        })
        for reward_gate in reward_gates[stage]:
            #reward_gate.sprite.visible = False
            general_reward_gates[stage]["x"].append(reward_gate.sprite.x)
            general_reward_gates[stage]["y"].append(reward_gate.sprite.y)
            general_reward_gates[stage]["x&width"].append(reward_gate.sprite.x + reward_gate.sprite.width)
            general_reward_gates[stage]["y&height"].append(reward_gate.sprite.y + reward_gate.sprite.height)

        
    #big_prize = shapes.Circle(450, 420, 20, color=(255, 255, 0), batch=batch)

def update_current_stage():
    for stage in range(len(obstacles)):
        for reward_gate in reward_gates[stage]:
            reward_gate.sprite.visible = (stage == current_stage)
        for obstacle in obstacles[stage]:
            obstacle.sprite.visible = (stage == current_stage)