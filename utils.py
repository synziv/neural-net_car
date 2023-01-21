import numpy as np
import pygame
import math
import mymap
vec2 = pygame.math.Vector2

def collides_with_obstacles(rocketX, rocketY, obstacles):
        #if(position[0] < 0 or position[0] > mymap.window_dimensions[0] or position[1] < 0 or position[1] > mymap.window_dimensions[1]):
        return np.any(
            (rocketX >= obstacles[mymap.current_stage]["x"]) &
            (rocketX <= obstacles[mymap.current_stage]["x&width"]) &
            (rocketY >= obstacles[mymap.current_stage]["y"]) & 
            (rocketY <= obstacles[mymap.current_stage]["y&height"])
        )
def collides_with_reward_gate(rocketX, rocketY, reward_gate_id):
    # print("x:", rocketX)
    # print("y:", rocketY)
    # print("reward_gate_id:", reward_gate_id)
    # print("obstacle x:", mymap.general_reward_gates["x"][reward_gate_id])
    # print("obstacle y:", mymap.general_reward_gates["y"][reward_gate_id])
    # print("obstacle x&width:", mymap.general_reward_gates["x&width"][reward_gate_id])
    # print("obstacle y&height:", mymap.general_reward_gates["y&height"][reward_gate_id])
    # print("rocketX >= mymap.general_reward_gates['x'][reward_gate_id]:", rocketX >= mymap.general_reward_gates["x"][reward_gate_id])
    # print("rocketX <= mymap.general_reward_gates['x&width'][reward_gate_id]:", rocketX <= mymap.general_reward_gates["x&width"][reward_gate_id])
    # print("rocketY >= mymap.general_reward_gates['y'][reward_gate_id]:", rocketY >= mymap.general_reward_gates["y"][reward_gate_id])
    # print("rocketY <= mymap.general_reward_gates['y&height'][reward_gate_id]:", rocketY <= mymap.general_reward_gates["y&height"][reward_gate_id])
    return np.any(
        (rocketX >= mymap.general_reward_gates[mymap.current_stage]["x"][reward_gate_id]) &
        (rocketX <= mymap.general_reward_gates[mymap.current_stage]["x&width"][reward_gate_id]) &
        (rocketY >= mymap.general_reward_gates[mymap.current_stage]["y"][reward_gate_id]) & 
        (rocketY <= mymap.general_reward_gates[mymap.current_stage]["y&height"][reward_gate_id])
    )
def collides_with_walls(rocketX, rocketY):
        return ((rocketX < 0) | (rocketX > mymap.window_dimensions[0]) | (rocketY < 0) | (rocketY > mymap.window_dimensions[1]))
def getCollisionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
        global vec2
        
        try:
            uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
            uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

            if 0 <= uA <= 1 and 0 <= uB <= 1:
                intersectionX = x1 + (uA * (x2 - x1))
                intersectionY = y1 + (uA * (y2 - y1))
                return vec2(intersectionX, intersectionY)
        except:
            #print("error")
            return None
        
        return None

def calculate_end_point(vision_line, old_angle, new_position):
    old_x = vision_line.line.x2 - vision_line.line.x
    old_y = vision_line.line.y2 - vision_line.line.y

    diff_angle = math.radians( -(vision_line.rotation - old_angle ))

    eol_x = ((old_x * math.cos(diff_angle) + old_y * math.sin(diff_angle)) + new_position[0])
    eol_y = ((-old_x * math.sin(diff_angle) + old_y * math.cos(diff_angle)) + new_position[1])

    return (eol_x, eol_y)

def collision_point_circle(point, circle):
    if point != []:
        if math.dist(point, circle.position) < circle.radius:
            return True
    return False

