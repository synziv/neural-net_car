import math
import time
from utils import calculate_end_point, getCollisionPoint
from vision_line import vision_line

import numpy as np
import pandas as pd
import mymap






mymap.initObstacle(None)
vs1 = {
    "x": [420.98, 100], 
    "y": [160.218, 100], 
    "x2": [221.311, 170], 
    "y2": [148.688, 250], 
    "length": [200, 100], 
    "angle": [-15, 100]
}
# vs1 = {
#     "x": np.full(10000, 100), 
#     "y": np.full(10000, 120), 
#     "x2": np.full(10000, 100 + 200), 
#     "y2": np.full(10000, 170), 
#     "length": np.full(10000, 200), 
#     "angle": np.full(10000, 200),
# }
df = pd.DataFrame(data=vs1)


x1 = df["x"].to_numpy()[:, None]
y1 = df["y"].to_numpy()[:, None]
x2 = df["x2"].to_numpy()[:, None]
y2 = df["y2"].to_numpy()[:, None]


# x3 = mymap.obstacles[0].df["x1"].values
# y3 = mymap.obstacles[0].df["y1"].values
# x4 = mymap.obstacles[0].df["x2"].values
# y4 = mymap.obstacles[0].df["y2"].values
x3 = mymap.obstaclesDf["x1"].values
y3 = mymap.obstaclesDf["y1"].values
x4 = mymap.obstaclesDf["x2"].values
y4 = mymap.obstaclesDf["y2"].values

start = time.time()
# obs = np.array([x1, y1, x2, y2])
# obs = np.resize(obs, (2, 4, 4))




uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))


intersectionX = np.where((0 <= uA) & (uA <= 1), x1 + (uA * (x2 - x1)), np.nan)
intersectionY = np.where((0 <= uB) & (uB <= 1), y1 + (uA * (y2 - y1)), np.nan)
print("intersectionX", intersectionX[ :])


test = np.stack((intersectionX, intersectionY), axis=2)

dist = np.sqrt(((intersectionX[:] - x1[0])**2 + (intersectionY[:] - y1[0])**2).astype(float))
#dist = np.sqrt(((test[:, :, 0] - x1[0])**2 + (test[:, :, 1] - y1[0])**2).astype(float))
# print("dist", dist)
min_row = np.nanargmin(dist, axis=1)
print("test\n", test)
coll_points = test[np.arange(len(test)), min_row]
print("test1\n", coll_points)
print("tito: ", intersectionX[np.arange(len(intersectionX)), min_row])
coll_points_X = intersectionX[np.arange(len(intersectionX)), min_row]
coll_points_Y = intersectionY[np.arange(len(intersectionY)), min_row]
print("test1\n", coll_points_X)
print("test1\n", coll_points_Y)


print("***************")


x = [(vision_line((100, 120), 15, 200, None, False))]
x[0].line.x = 420.98
x[0].line.y = 160.218
x[0].line.x2 = 221.311
x[0].line.y2 = 148.688




def check_colisions_obstacles(v_l):
    #calculer la dist la plus courte pour afficher le point le plus proche slm
    is_collision = False
    smallest_dist = 1000
    end = (v_l.line.x2, v_l.line.y2)
    origin = (v_l.line.x, v_l.line.y)

    best_collision_point = 0
    dist = 0
    # print("origin\n", origin)
    # print("end\n", end)
    for obstacle in mymap.obstacles:
        for line in obstacle.obstacle_lines:
            
            collision_point = getCollisionPoint(origin[0], origin[1], end[0], end[1], line[0][0], line[0][1], line[1][0],line[1][1])
            if(collision_point != None):
                dist = math.dist(origin, collision_point)
                is_collision = True
                if(dist < smallest_dist):
                    smallest_dist = dist
                    best_collision_point = collision_point
                    v_l.collision_point_sprite.visible = v_l.show_vision_line
                    v_l.collision_point_sprite.position =  collision_point
                    v_l.collision_point = collision_point
            elif not is_collision:
                v_l.collision_point_sprite.visible = False
                v_l.collision_point = [-1, -1]
    print("dist\n", smallest_dist)
    print("collision_point\n", best_collision_point)
    print("--------------------------")
    return is_collision

start = time.time()
for(v_l) in x:
    check_colisions_obstacles(v_l)

#print("Version naive: ", time.time() - start)

# nb = range(2)

# col = ["x", "y", "x2", "y2", "length", "angle"]

# multi = pd.MultiIndex.from_product([nb, col], names=["nb", "col"])
# df = pd.DataFrame(np.array([[100, 100, 100 + 200, 100, 200, 76] * 2]*11), 
#                             columns=multi,
#                             index=range(11))

# print(df)
# df[0, "x"] = 420.98
# print(df)

rockets = np.vstack([[pd.DataFrame(data={
            "x": [100], 
            "y": [100], 
            "rotation": [0],
            "x_speed": [0], 
            "y_speed": [0], 
            "acceleration": [0],
            "points": [0],
            "life": [0],
            "current_reward_gate_id": [0],
        })]]*2)

print(rockets)
#print(rockets[:, 0])