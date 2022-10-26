from pyglet import shapes
import math
import pygame
from pyglet.window import key
from vision_line import vision_line
from utils import collides_with, collision_point_circle
import mymap
import numpy as np
from fastbook import *
import copy



vec2 = pygame.math.Vector2
MAX_ACCELERATION = 400


class rocket:
    def __init__(self, batch, brain, show_vision_lines = True, ):
        self.position = (100, 100)
        self.x_speed =0
        self.y_speed =0
        self.rotation = 0
        self.rotate_speed = 5.0
        self.acceleration =0
        self.points =0
        self.show_vision_lines = show_vision_lines
        self.sprite = shapes.Rectangle(0, 0, 30, 60, color=(255, 22, 20), batch=batch)
        self.sprite.anchor_x = self.sprite.width / 2
        self.sprite.anchor_y = self.sprite.height / 2
        self.status = "alive"
        self.fitness= 0
        self.life = 0
        self.life_reward = 0
        self.brain = brain
        self.current_reward_gate_id = 0

        self.vision_lines = [
            vision_line(self.position, 0, batch, self.show_vision_lines),
            vision_line(self.position, -15, batch, self.show_vision_lines),
            vision_line(self.position, 15, batch, self.show_vision_lines), 
            vision_line(self.position, -30, batch, self.show_vision_lines),
            vision_line(self.position, 30, batch, self.show_vision_lines), 
            vision_line(self.position, -60, batch, self.show_vision_lines),
            vision_line(self.position, 60, batch, self.show_vision_lines),
            vision_line(self.position, -90, batch, self.show_vision_lines),
            vision_line(self.position, 90, batch, self.show_vision_lines)
        ]
        self.data = torch.tensor([self.position[0], self.position[1], self.rotation, self.points, *[0] * len(self.vision_lines)*2])

        
    
    def getControls(self, dt):
    #def getControls(self, dt, keys):
        output = self.brain.read_outputs(self.data)
        if output == 0:
            self.rotation = self.rotation + dt * self.rotate_speed
            
        if output == 1:
            self.rotation = self.rotation - dt * self.rotate_speed

        if output == 2:
            if(self.acceleration < MAX_ACCELERATION):
                self.acceleration += 25
        else:
            #get slower if the user is not pressing the up key
            if(self.acceleration > 0):
                self.acceleration -= 10

        #controls for human player
        # if keys[key.LEFT]:
        #     self.rotation = self.rotation + dt * self.rotate_speed

            
        # if keys[key.RIGHT]:
        #     self.rotation = self.rotation - dt * self.rotate_speed
            

        # if keys[key.UP]:
        #     if(self.acceleration < MAX_ACCELERATION):
        #         self.acceleration += 25
        # else:
        #     #get slower if the user is not pressing the up key
        #     if(self.acceleration > 0):
        #         self.acceleration -= 10
                
        

        
    def crossover(self, partner):
        self.brain.crossover(partner.brain)
        
    def update(self, dt):
    #def update(self,dt, keys):
        #self.getControls(dt, keys)
        self.getControls(dt)

        self.x_speed = math.cos(self.rotation) * self.acceleration
        self.y_speed = math.sin(self.rotation) * self.acceleration   
        self.position = (self.position[0] + dt * self.x_speed, self.position[1] + dt * self.y_speed)

        self.sprite.rotation = 90 - math.degrees(self.rotation)
        self.sprite.x = self.position[0]
        self.sprite.y = self.position[1]

        rotation_degree = math.degrees(self.rotation)

        for vision_line in self.vision_lines:
            vision_line.update(rotation_degree, self.position)
        
        for obstacle in mymap.obstacles:
            if(collides_with(self.sprite, obstacle.sprite)):
                self.status = "dead"
                self.points = self.points * 0.95
                #print("dead by obstacle")


        if(self.current_reward_gate_id < len(mymap.reward_gates) and 
            collides_with(self.sprite, mymap.reward_gates[self.current_reward_gate_id].sprite)):
            self.points += 25 +  100 / ((self.life_reward) * 0.2)
            #print(self.points)
            self.current_reward_gate_id += 1
            self.life_reward = 0
            #mymap.reward_gates.pop(0)
            #print("woohoo")

        if(collision_point_circle(self.position, mymap.big_prize)):
            #print("big prize")
            self.status = "won"
            self.points += 100+  1000 / (self.life_reward * 0.2)
        if(collision_wall(self.position)):
            self.status = "dead"
            self.points = self.points * 0.95
            #print("dead by wall")
        if(self.life > 200):
            self.status = "dead"
            #print("dead by life")
        if(self.status == "dead" ):
            self.sprite.color = (255, 255, 255)
            self.points += 100 / self.calculate_shortest_distance_to_reward()
            #print("distance: ", self.calculate_shortest_distance_to_reward())
            #print("points: ", self.points)
        else:
            self.life += 1
            self.life_reward += 1
            self.sprite.color = (255, 22, 20)
        
        self.calculate_shortest_distance_to_reward()
        self.data = torch.tensor([self.position[0], self.position[1], self.rotation, self.points, *self.flattened_collision_points()])
    
    def flattened_collision_points(self):
        collisions = np.array([])
        for o in self.vision_lines:
            collisions = np.append(collisions, o.collision_point)
        return collisions
    
    def reset(self):
        self.position = (100, 100)
        self.x_speed =0
        self.y_speed =0
        self.rotation = 0
        self.acceleration =0
        self.points =0
        self.sprite.position = self.position
        self.status = "alive"
        self.fitness= 0
        self.life = 0
        self.life_reward = 0
        self.current_reward_gate_id = 0


    def calculate_shortest_distance_to_reward(self):
        if(self.current_reward_gate_id < len(mymap.reward_gates)):
            return self.calculate_shortest_distance_to_reward_gate()
        if(self.current_reward_gate_id == len(mymap.reward_gates)):
            return math.sqrt((self.position[0] - mymap.big_prize.x)**2 + (self.position[1] - mymap.big_prize.y )**2)

    def calculate_shortest_distance_to_reward_gate(self):
        current_gate = mymap.reward_gates[self.current_reward_gate_id]
        shortest_distance = 100000
        #depending on the angle of the reward gate, calculate shortest distance
        #using multiple points with interval of 20 pixels
        #print("Current reward gate: ", self.current_reward_gate_id)
        if(current_gate.rotation == 0):
            upper_limit = current_gate.sprite.width
        else:
            upper_limit = current_gate.sprite.height

        for possible_col in range(0, upper_limit, 20):
            #print("rotation: ", current_gate.rotation)
            if(current_gate.rotation == 0):
               
                distance = math.sqrt((self.position[0] - (current_gate.sprite.x + possible_col))**2 + (self.position[1] - current_gate.sprite.y )**2)
            else:
            # print((self.current_reward_gate.sprite.x , self.current_reward_gate.sprite.y+ possible_col))
            # print(self.position)
            # self.short = shapes.Line(self.position[0], self.position[1], self.current_reward_gate.sprite.x, self.current_reward_gate.sprite.y + possible_col, color=(255, 255, 255), batch=self.batch)
                distance = math.sqrt((self.position[0] - current_gate.sprite.x)**2 + (self.position[1] - (current_gate.sprite.y + possible_col))**2)
        
            if(distance < shortest_distance):
                shortest_distance = distance
        return shortest_distance


def sigmoid(x):
  return 1 / (1 + math.e ** -x)

def collision_wall(position):
    if(position[0] < 0 or position[0] > mymap.window_dimensions[0] or position[1] < 0 or position[1] > mymap.window_dimensions[1]):
        return True
    return False