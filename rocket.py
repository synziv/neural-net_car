from pickle import FALSE
from pyglet import shapes
import math
import pygame
from pyglet.window import key
from vision_line import vision_line
from utils import collides_with
import map


vec2 = pygame.math.Vector2
MAX_ACCELERATION = 400


class rocket:
    def __init__(self, batch):
        self.position = (100, 100)
        self.x_speed =0
        self.y_speed =0
        self.rotation = 0
        self.rotate_speed = 5.0
        self.acceleration =0
        self.sprite = shapes.Rectangle(0, 0, 30, 60, color=(255, 22, 20), batch=batch)
        self.sprite.anchor_x = self.sprite.width / 2
        self.sprite.anchor_y = self.sprite.height / 2
        

        # self.vision_lines = [
        #     (shapes.Rectangle(self.position[0], self.position[1], 2, 400, color=(255, 22, 20), batch=batch), 0),
        #     (shapes.Rectangle(self.position[0], self.position[1], 2, 400, color=(255, 22, 20), batch=batch), 15),
        #     (shapes.Rectangle(self.position[0], self.position[1], 2, 400, color=(255, 22, 20), batch=batch), -15)
        # ]
        #self.circle = shapes.Circle(self.position[0] + 250, self.position[1], 5, color=(255, 255, 255), batch=batch)
        #self.collision = shapes.Circle(0, 0, 5, color=(255, 255, 255), batch=batch)

        self.vision_lines = [
            vision_line(self.position, 0, batch),
            vision_line(self.position, -15, batch),
            vision_line(self.position, 15, batch), 
            vision_line(self.position, -30, batch),
            vision_line(self.position, 30, batch), 
            vision_line(self.position, -60, batch),
            vision_line(self.position, 60, batch),
            vision_line(self.position, -90, batch),
            vision_line(self.position, 90, batch)
        ]
        
    

    def getControls(self, dt, keys):
        # if(self.test < 10):
        #     self.rotation = self.rotation + dt * self.rotate_speed
        #     self.test = self.test + 1
        if keys[key.LEFT]:
            self.rotation = self.rotation + dt * self.rotate_speed

            
        if keys[key.RIGHT]:
            self.rotation = self.rotation - dt * self.rotate_speed
            

        if keys[key.UP]:
            if(self.acceleration < MAX_ACCELERATION):
                self.acceleration += 25
        else:
            #get slower if the user is not pressing the up key
            if(self.acceleration > 0):
                self.acceleration -= 10


        # old_x = self.circle.x - self.position[0]
        # old_y = self.circle.y - self.position[1]
        

        

        self.x_speed = math.cos(self.rotation) * self.acceleration
        self.y_speed = math.sin(self.rotation) * self.acceleration   
        self.position = (self.position[0] + dt * self.x_speed, self.position[1] + dt * self.y_speed)

        self.sprite.rotation = 90 - math.degrees(self.rotation)
        self.sprite.x = self.position[0]
        self.sprite.y = self.position[1]

        rotation_degree = math.degrees(self.rotation)

        for vision_line in self.vision_lines:
            vision_line.update(rotation_degree, self.position)
        
        collision = False
        for obstacle in map.obstacles:
            if(collides_with(self.sprite, obstacle.sprite)):
                collision = True
        if(collision):
            self.sprite.color = (255, 255, 255)
        else:
            self.sprite.color = (255, 22, 20)