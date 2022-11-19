import pygame
import math
vec2 = pygame.math.Vector2

def collides_with(self, other_object):

        if (self.x >= other_object.x and
            self.x <= other_object.x + other_object.width and
            self.y >= other_object.y and
            self.y <= other_object.y + other_object.height):
            return True
        else:
            return False

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

