from pyglet import shapes
import mymap
from utils import getCollisionPoint
import math

from utils import calculate_end_point

class vision_line:
    def __init__(self, origin, origin_angle,length, batch, show_vision_line):
        self.rotation = 0
        self.origin_angle = -origin_angle
        self.line = shapes.Line(origin[0], origin[1], origin[0]+length, origin[1], color=(255, 22, 20), batch=batch)
        self.collision_point = []
        self.collision_dist = 200
        self.length = length
        self.collision_point_sprite = shapes.Circle(0, 0, 5, color=(255, 255, 255), batch=batch)

        x2, y2 = calculate_end_point(self, self.origin_angle, origin)
        self.line.x2 = x2
        self.line.y2 = y2
        self.show_vision_line = show_vision_line
        self.line.visible = show_vision_line
        self.batch = batch

    def update(self, new_angle, new_position):
        old_angle = self.rotation
        self.rotation = new_angle

        x2, y2 = calculate_end_point(self, old_angle, new_position)

        self.line.x = new_position[0]
        self.line.y = new_position[1]
        self.line.x2 = x2
        self.line.y2 = y2

        self.check_collisions()
    

    def check_collisions(self):
        collided_with_obstacles = self.check_colisions_obstacles()
        
        if(not collided_with_obstacles):
            self.check_collisions_window()

    def check_colisions_obstacles(self):
        #calculer la dist la plus courte pour afficher le point le plus proche slm
        is_collision = False
        smallest_dist = 1000
        end = (self.line.x2, self.line.y2)
        origin = (self.line.x, self.line.y)

        collision_point = None
        for obstacle in mymap.obstacles:
            for line in obstacle.obstacle_lines:
                collision_point = getCollisionPoint(origin[0], origin[1], end[0], end[1], line[0][0], line[0][1], line[1][0],line[1][1])
                if(collision_point != None):
                    dist = math.dist(origin, collision_point)   
                    #print("origin\n", origin)
                    #print("end\n", end)         
                    is_collision = True
                    if(dist < smallest_dist):
                        smallest_dist = dist
                        self.collision_point_sprite.visible = self.show_vision_line
                        self.collision_point_sprite.position =  collision_point
                        self.collision_point = collision_point
                elif not is_collision:
                    self.collision_point_sprite.visible = False
                    self.collision_point = [-1, -1]
        #print(self.collision_point)
        return is_collision
    
    def check_collisions_window(self):
        origin = (self.line.x, self.line.y)
        is_collision = False

        for line in mymap.window_lines:
            collision_point = getCollisionPoint(origin[0], origin[1], self.line.x2, self.line.y2, line[0][0], line[0][1], line[1][0],line[1][1])
            if(collision_point != None):
                self.collision_point_sprite.position =  collision_point
                self.collision_point = collision_point
                is_collision = True

        if not is_collision:
            self.collision_point_sprite.visible = False
            self.collision_point = [-1, -1]
