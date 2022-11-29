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


class Population:
    def __init__(self, num_of_agents, batch):
        #self.agents = [rocket(batch, AI(6, 20, 4), True) for agent in range(num_of_agents)]
        self.pop_size = num_of_agents
        self.alive_agents = num_of_agents
        self.calculating = False
        self.batch = batch
        self.all_visual_lines = np.array([])
        self.rockets = np.array([])

        self.init_vision_lines()
        self.init_rockets()


        self.rocket_sprites = np.full(self.pop_size, shapes.Rectangle(0, 0, 30, 60, color=(255, 22, 20), batch=batch))





    
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
        ]]*self.pop_size)

        print(self.rockets)

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
        v_l = [[
            100, #x
            100, #y
            100 + 200, # x1 #origine + length of vision line
            100, #x2
            200, #length
        ] ]* 11
        ag = np.arange(-75, 76, 15).reshape(-1 ,1)

        v_l = np.append(v_l, ag, axis=1)

        self.all_visual_lines = np.array([v_l] * self.pop_size)

        #self.calculate_vision_lines()
        #self.update_vision_lines()
        




    def update_rockets(self):
        self.rockets[:, 4] += 1
        print(self.rockets)
        self.rockets["x_speed"] += np.cos(np.radians(self.rockets["rotation"])) * self.rockets["acceleration"]
        self.rockets["y_speed"] += np.sin(np.radians(self.rockets["rotation"])) * self.rockets["acceleration"]
        self.rockets["x"] += self.rockets["x_speed"]
        self.rockets["y"] += self.rockets["y_speed"]



        #render rockets
        for i in range(self.pop_size):
            self.rocket_sprites[i].x = self.rockets[i]["x"]
            self.rocket_sprites[i].y = self.rockets[i]["y"]
            self.rocket_sprites[i].rotation = self.rockets[i]["rotation"]









    
    def calculate_vision_lines(self, rotation_degree = 0):
        print("angle: ", self.all_visual_lines[:,:,5])
        old_angle = self.all_visual_lines[:,:,5].copy()

        self.all_visual_lines[:,:,5] = rotation_degree

        old_x = self.all_visual_lines[:,:,2] - self.all_visual_lines[:,:,0] #x1-x
        old_y = self.all_visual_lines[:,:,3] - self.all_visual_lines[:,:,1] #y1-y

        diff_angle = np.radians( -(self.all_visual_lines[:,:,5] - old_angle ))

        self.all_visual_lines[:,:,2] = ((old_x * np.cos(diff_angle) + old_y * np.sin(diff_angle)) + self.position[0])
        self.all_visual_lines[:,:,3] = ((-old_x * np.sin(diff_angle) + old_y * np.cos(diff_angle)) + self.position[1])

        self.all_visual_lines["x"] = self.position[0]
        self.all_visual_lines["y"] = self.position[1]

        #show vision_lines
        # for rocket_i in range(self.rockets):
        #     for line_i in range(self.all_visual_lines[rocket_i]):
        #         self.all_visual_lines[rocket_i][line_i].set_data([self.all_visual_lines[rocket_i][line_i]["x"], self.all_visual_lines[rocket_i][line_i]["x2"]], [self.all_visual_lines[rocket_i][line_i]["y"], self.all_visual_lines[rocket_i][line_i]["y2"]])










    def update_vision_lines(self):
        x1 = self.all_visual_lines[:,:,0]
        y1 = self.all_visual_lines[:,:,1]
        x2 = self.all_visual_lines[:,:,2]
        y2 = self.all_visual_lines[:,:,3]
        
        x3 = mymap.obstaclesDf["x1"].values
        y3 = mymap.obstaclesDf["y1"].values
        x4 = mymap.obstaclesDf["x2"].values
        y4 = mymap.obstaclesDf["y2"].values





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
        