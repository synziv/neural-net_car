import array
import copy
import numpy as np
import pandas as pd
import torch
from ai import AI
from rocket import rocket
import random
import mymap


class Population:
    def __init__(self, num_of_agents, batch):
        self.agents = [rocket(batch, AI(6, 20, 4), True) for agent in range(num_of_agents)]
        self.pop_size = num_of_agents
        self.alive_agents = num_of_agents
        self.calculating = False
        self.batch = batch
        self.all_visual_lines = np.array([])

        self.init_vision_lines()
    
    def update(self, dt):
        self.alive_agents = 0
        for agent in self.agents:
            if(agent.status == "alive"):
                self.alive_agents += 1
                #print(alive_agents)
                agent.update(dt)
        
    def init_vision_lines(self):
        self.all_visual_lines = np.vstack([[pd.DataFrame(data={
            "x": np.full(11, 100), 
            "y": np.full(11, 100), 
            "x2": np.full(11, 100 + 200), #origine + length of vision line
            "y2": np.full(11, 100), 
            "length": np.full(11, 200), 
            "angle": np.arange(-75, 76, 15),
        })]]*self.pop_size)
        # a = [[1, 2, 3], [4, 5, 6]]
        # b = [[7, 8, 9], [10, 11, 12]]
        # c = np.vstack([a, b])
        self.update_vision_lines()
        print(self.all_visual_lines)


    
    def calculate_vision_lines(self, rotation_degree = 0):
        old_angle = self.vision_lines1["angle"].copy()
        
        self.vision_lines1["angle"] = rotation_degree

        old_x = self.vision_lines1["x2"] - self.vision_lines1["x"]
        old_y = self.vision_lines1["y2"] - self.vision_lines1["y"]

        diff_angle = np.radians( -(self.vision_lines1["angle"] - old_angle ))

        self.vision_lines1["x2"] = ((old_x * np.cos(diff_angle) + old_y * np.sin(diff_angle)) + self.position[0])
        self.vision_lines1["y2"] = ((-old_x * np.sin(diff_angle) + old_y * np.cos(diff_angle)) + self.position[1])

        self.vision_lines1["x"] = self.position[0]
        self.vision_lines1["y"] = self.position[1]











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
            

    # mutation = () => {
    #   for(let i =0; i< this.genes.length; i++){
    #     if(Math.random()< 0.01 ){
    #       const mutationChoice = Math.random();
    #       if(mutationChoice< 0.33)
    #         this.genes[i] = p5.Vector.random2D();
    #       else if(mutationChoice< 0.66){
    #         //this.genes[i] = p5Glob.createVector(0,0);
    #         if(i != 0 && i !=this.genes.length-1){
    #           const v1 = this.genes[i-1].copy();
    #           const v2 = this.genes[i+1].copy();
    #           const v = v1.add(v2).mult(0.5);
    #           this.genes[i] = v;
              
    #         }
            
    #       }
    #       else
    #         this.genes[i].mult(10);
    #       this.genes[i].setMag(GlobalVar.maxForce);
    #     }
    #   }
    # }