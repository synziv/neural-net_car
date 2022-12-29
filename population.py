import array
import copy
import numpy as np
import pandas as pd
import torch
from ai import AI
from rocket import rocket
import random
import mymap
from pyglet import shapes
import math
import time

from utils import collides_with_obstacles, collides_with_reward_gate, collides_with_walls, collision_point_circle

MAX_LIFE = 300


class Population:
    def __init__(self, num_of_agents, batch):
        #self.agents = [rocket(batch, AI(6, 20, 4), True) for agent in range(num_of_agents)]
        self.pop_size = num_of_agents
        self.alive_agents = num_of_agents
        self.calculating = False
        self.batch = batch
        self.all_visual_lines = np.array([])
        self.rockets = np.array([])
        self.render_v_l = []
        self.rocket_sprites = [shapes.Rectangle(0, 0, 30, 60, color=(255, 22, 20), batch=batch) for i in range(num_of_agents)]
        
        self.init_rockets()
        self.init_vision_lines()
        self.init_brains()







    
    def update(self, dt):
        self.alive_agents = 0
        for agent in self.agents:
            if(agent.status == "alive"):
                self.alive_agents += 1
                #print(alive_agents)
                agent.update(dt)
    
    def init_rockets(self):
        self.rockets = np.array([[
            100, #x 0
            100, #y 1
            0,   #x_speed 2
            0,   #y_speed 3 
            0,   #rotation 4
            0,   #acceleration 5
            0,   #life 6
            0,   #points 7
            0,   #reward_gate_id 8
            0,   #is_dead
            0,   #life_reward
        ]]*self.pop_size)
        self.rockets = {
            'x' : self.rockets[:, 0].astype(np.float32),
            'y' : self.rockets[:, 1].astype(np.float32),
            'x_speed' : self.rockets[:, 2].astype(np.float32),
            'y_speed' : self.rockets[:, 3].astype(np.float32),
            'rotation' : self.rockets[:, 4].astype(np.float32),
            'acceleration' : self.rockets[:, 5].astype(np.float32),
            'life' : self.rockets[:, 6],
            'points' : self.rockets[:, 7].astype(np.float32),
            'reward_gate_id' : self.rockets[:, 8],
            'is_dead' : self.rockets[:, 9].astype(np.float32),
            'life_reward' : self.rockets[:, 10].astype(np.float32),
        }
        self.rocketsData = [
            torch.tensor([
                self.rockets["x"][0], 
                self.rockets["y"][0], 
                self.rockets["rotation"][0], 
                self.rockets["points"][0], 
                *[0] * 11 * 2 # 11 vision lines, 2 values for each (x, y)
            ])
        ]* self.pop_size

        for i in range(self.pop_size):
            self.rocket_sprites[i].x = self.rockets["x"][i]
            self.rocket_sprites[i].y = self.rockets["y"][i]
            self.rocket_sprites[i].anchor_x = 15
            self.rocket_sprites[i].anchor_y = 30
            self.rocket_sprites[i].rotation = 75
            self.rocket_sprites[i].rotation = self.rockets["rotation"][i]



        # self.rockets = np.vstack([[pd.DataFrame(data={
        #     "x": [100], 
        #     "y": [100], 
        #     "rotation": [0],
        #     "x_speed": [0], 
        #     "y_speed": [0], 
        #     "acceleration": [0],
        #     "points": [0],
        #     "life": [0],
        #     "current_reward_gate_id": [0],
        # })]]*self.pop_size)
        
    def init_vision_lines(self):
        # df = pd.DataFrame(data={
        #     "x": np.full(11, 100), 
        #     "y": np.full(11, 100), 
        #     "x2": np.full(11, 100 + 200), #origine + length of vision line
        #     "y2": np.full(11, 100), 
        #     "length": np.full(11, 200), 
        #     "angle": np.arange(-75, 76, 15),
        # })
        # self.all_visual_lines = np.vstack([df.columns, df.to_numpy()]*self.pop_size)

        # nb = range(self.pop_size)

        # col = ["x", "y", "x2", "y2", "length", "angle"]

        # multi = pd.MultiIndex.from_product([nb, col], names=["nb", "col"])
        # self.all_visual_lines = pd.DataFrame(np.array([[100, 100, 100 + 200, 100, 200, 76] * self.pop_size]*11), #11 is the nb of vision lines per rockets
        #                             columns=multi,
        #                             index=range(11))  

        center_x = self.rockets['x'][0] + (30 / 2)
        center_y = self.rockets['y'][0] + (60 / 2)
        v_l = [[
            center_x, #x
            center_y, #y
            100 + 200, # x1 #origine + length of vision line
            100, #x2
            200, #length
        ] ]* 11
        ag = np.arange(-75, 76, 15).reshape(-1 ,1)

        v_l = np.append(v_l, ag, axis=1)

        self.all_visual_lines = np.array([v_l] * self.pop_size)
        self.all_visual_lines = {
            'x' : self.all_visual_lines[:, :, 0].astype(np.float32),
            'y' : self.all_visual_lines[:, :, 1].astype(np.float32),
            'x2' : self.all_visual_lines[:, :, 2].astype(np.float32),
            'y2' : self.all_visual_lines[:, :, 3].astype(np.float32),
            'length' : self.all_visual_lines[:, :, 4].astype(np.float32),
            'angle' : self.all_visual_lines[:, :, 5].astype(np.float32),
        }

        self.render_v_l = [[
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(200,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch),
            shapes.Line(center_x, center_y, center_x + 200, 0, color=(30,144,255), batch=self.batch)
            ] for _ in range(self.pop_size)
        ]
        self.render_col_points = [[
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch),
            shapes.Circle(center_x, center_y, 5, color=(255, 255, 255), batch=self.batch)
            ] for _ in range(self.pop_size)
        ]

        self.calculate_vision_lines()
        #self.update_vision_lines()
        





    def update_rockets(self):
        start = time.time()

        self.rockets['acceleration'][self.rockets['acceleration'] < 5] += np.random.rand() * 0.1
        self.rockets['rotation'] += np.random.randint(0, 2, self.pop_size)
        
        self.rockets["x_speed"] = np.cos(np.radians(self.rockets["rotation"])) * self.rockets["acceleration"]
        self.rockets["y_speed"] = np.sin(np.radians(self.rockets["rotation"])) * self.rockets["acceleration"]
        self.rockets["x"] += self.rockets["x_speed"]
        self.rockets["y"] += self.rockets["y_speed"]

        self.calculate_vision_lines()
        collisions = self.check_collisions_vision_lines()
        #end = time.time()
        #print("vector: ", end - start)

        #start = time.time()
        #render rockets
        for i in range(self.pop_size):
            if(self.rockets["life"][i] < MAX_LIFE and self.rockets["is_dead"][i] == 0):
                #check collision with obstacles
                if(collides_with_obstacles(self.rockets["x"][i], self.rockets["y"][i], mymap.general_obstacles)):
                    self.rockets["points"][i] *= 0.95
                    self.rocket_sprites[i].color = (255, 255, 255)
                    self.rockets["is_dead"][i] = 1
                #if no collision with obstacles, check collision with walls
                elif(collides_with_walls(self.rockets["x"][i], self.rockets["y"][i])):
                    self.rockets["points"][i] *= 0.95
                    self.rocket_sprites[i].color = (255, 255, 255)
                    self.rockets["is_dead"][i] = 1
                    
                #check collision with reward gates
                if(self.rockets["reward_gate_id"][i] < len(mymap.reward_gates) and 
                    collides_with_reward_gate(self.rockets["x"][i], self.rockets["y"][i], self.rockets["reward_gate_id"][i])):
                    self.rockets["points"][i] += 25 +  100 / ((self.rockets["life_reward"][i]) * 0.2)
                    self.rockets["reward_gate_id"] += 1 
                    self.rockets["life_reward"][i] = 0
                #check collision with big prize
                elif(collision_point_circle((self.rockets["x"][i], self.rockets["y"][i]), mymap.big_prize)):
                    self.rockets["points"][i] += 100+  1000 / (self.life_reward * 0.2)

                self.rockets["life"][i] += 1
                self.rockets["life_reward"][i] += 1
                #for rendering
                self.rocket_sprites[i].x = self.rockets["x"][i]
                self.rocket_sprites[i].y = self.rockets["y"][i]
                self.rocket_sprites[i].rotation = 90 - self.rockets["rotation"][i]

                #update data
                self.rocketsData[i] = torch.tensor([
                    self.rockets["x"][i], 
                    self.rockets["y"][i], 
                    self.rockets["rotation"][i], 
                    self.rockets["points"][i], 
                    *collisions["x"][i],
                    *collisions["y"][i],
                ])
            else:
                self.rockets["is_dead"][i] = 1
        end = time.time()
        print("for-loop: ", end - start)





    
    def calculate_vision_lines(self, rotation_degree = 0):

        old_angle = self.all_visual_lines['angle'].copy()

        self.all_visual_lines['angle'] = self.rockets['rotation'][:, None].copy()

        old_x = self.all_visual_lines['x2'] - self.all_visual_lines['x']
        old_y = self.all_visual_lines['y2'] - self.all_visual_lines['y']

        diff_angle = np.radians( -(self.all_visual_lines['angle'] - old_angle ))

        self.all_visual_lines['x2'] = ((old_x * np.cos(diff_angle) + old_y * np.sin(diff_angle)) + self.rockets['x'][:, None])
        self.all_visual_lines['y2'] = ((-old_x * np.sin(diff_angle) + old_y * np.cos(diff_angle)) + self.rockets['y'][:, None])

        self.all_visual_lines['x'] = self.rockets['x'][:, None].copy()
        self.all_visual_lines['y'] = self.rockets['y'][:, None].copy()


        #show vision_lines
        # for rocket_i in range(self.pop_size):
        #     for line_i in range(len(self.all_visual_lines['x2'][rocket_i])):
        #         self.render_v_l[rocket_i][line_i].x = self.all_visual_lines['x'][rocket_i]
        #         self.render_v_l[rocket_i][line_i].y = self.all_visual_lines['y'][rocket_i]
        #         self.render_v_l[rocket_i][line_i].x2 = self.all_visual_lines['x2'][rocket_i, line_i]
        #         self.render_v_l[rocket_i][line_i].y2 = self.all_visual_lines['y2'][rocket_i, line_i]



    #Calculate vision lines collision points with vectorization
    #return a dict of x and y values of collision points
    def check_collisions_vision_lines(self):
        #start = time.time()
        
        x1 = self.all_visual_lines["x"][:, None]
        y1 = self.all_visual_lines["y"][:, None]
        x2 = self.all_visual_lines["x2"][:, None]
        y2 = self.all_visual_lines["y2"][:, None]

        #start = time.time()
        x3 = mymap.obstaclesDf["x1"].values[:, None]
        y3 = mymap.obstaclesDf["y1"].values[:, None]
        x4 = mymap.obstaclesDf["x2"].values[:, None]
        y4 = mymap.obstaclesDf["y2"].values[:, None]
        


        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        
        
        #calulate where the collision points are
        intersectionX = np.where((0 <= uA) & (uA <= 1), x1 + (uA * (x2 - x1)), np.inf)
        intersectionY = np.where((0 <= uB) & (uB <= 1), y1 + (uA * (y2 - y1)), np.inf)

        #calculate the distance between the car and the collision point for each vision line
        dist = np.sqrt(((intersectionX[:] - x1[0])**2 + (intersectionY[:] - y1[0])**2).astype(float))

        #get the minimum distance row index
        min_row = np.nanargmin(dist, axis=1)

        intersectionX = np.swapaxes(intersectionX, 1, 2)
        intersectionY = np.swapaxes(intersectionY, 1, 2)


        #get the collision points depending on the closest edge the vision line hits
        m,n = min_row.shape
        m = np.arange(m)[:,None]
        n = np.arange(n)

        coll_points_X = intersectionX[m, n, min_row]
        coll_points_Y = intersectionY[m, n, min_row]

        return {
            "x": coll_points_X, 
            "y": coll_points_Y
        }
        

        #showing collision points
        # for rocketI in range(self.pop_size):
        #     for i in range(coll_points_X.shape[1]):
        #         self.render_col_points[rocketI][i].x = coll_points_X[rocketI][i]
        #         self.render_col_points[rocketI][i].y = coll_points_Y[rocketI][i]
                
        #         if(np.isnan(coll_points_X[rocketI][i]) or np.isnan(coll_points_Y[rocketI][i])):
        #             self.render_col_points[rocketI][i].visible = False
        #         else:
        #             self.render_col_points[rocketI][i].visible = True





    def selection(self):
        #print(len(self.agents))
        self.calculating = True
        newPopulation = []
        self.agents.sort(key=lambda x: x.points, reverse=True)

        #only keep the best 40%
        self.agents = self.agents[:int(len(self.agents) * 0.6)]


        # for agent in self.agents:
        #     print(agent.points)


        #normalize fitness
        self.calc_fitness()

        nb_of_elites = self.elitismSelection(0.2, newPopulation)
        #print("nb of elites", nb_of_elites)
        #the last added individual is the best of last gen
        #make it show
#     newPopulation[newPopulation.length - 1].elite = true;
        #newPopulation[-1].elite = True
#     this.crossover(newPopulation, nbOfElites);

#     this.rockets = newPopulation;
#     GlobalVar.livingRockets = this.popSize;

        self.cross_over(newPopulation, nb_of_elites)
        self.mutate(newPopulation, nb_of_elites)
        self.agents = newPopulation
        self.calculating = False
        self.reset()

    def calc_fitness(self):
        max = self.agents[0].points
        min = self.agents[-1].points

        print("max:", max, "min:", min)

        for agent in self.agents:
            #print("points: ",agent.points)
            agent.fitness = (agent.points - min) / (max - min)
            #print("fitness: ",agent.fitness)
            #print("------------------")


    def elitismSelection(self, elitismRate, newPopulation):
        nbOfElites = int(elitismRate * self.pop_size)
        for i in range(nbOfElites):
            newPopulation.append(rocket(self.batch, AI(22, 16, 4, self.agents[i].brain.simple_net), False))
        #print(self.agents[0].brain.simple_net[0].weight)
        #print(newPopulation[0].brain.simple_net[0].weight)
        return nbOfElites

    def cross_over(self, newPopulation, nb_of_elites):
        for i in range(self.pop_size - nb_of_elites):
            parent1 = self.pickRandom(self.agents)
            #print("parent1", parent1.fitness)
            parent2 = self.pickRandom(self.agents)
            #child = copy.deepcopy(parent1)
            child_brain = AI(22, 16, 4, parent1.brain.simple_net)
            child_brain.crossover(parent2.brain)
            child = rocket(self.batch, child_brain, False)
            newPopulation.append(child)
    
    def pickRandom(self, matingPool):
        pickedParent = None
        while pickedParent == None:
            randomParent = random.randint(0, len(matingPool)-1)
            randomPickChance = random.randint(0, 100) / 100
            pickedParent = matingPool[randomParent]
            #print(pickedParent.fitness, randomPickChance, pickedParent.fitness < randomPickChance)
            if pickedParent.fitness < randomPickChance:
                pickedParent = None
        #print("picked: ", pickedParent.points)
        return pickedParent



    def reset(self):
        self.alive_agents = self.pop_size
    
    
    def mutate(self, newPopulation, nb_of_elites):
        for i in range(int(nb_of_elites/2), len(newPopulation)):
            newPopulation[i].brain.mutate(0.01)
        