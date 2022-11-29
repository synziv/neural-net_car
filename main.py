import pyglet
from pyglet.gl import *

from pyglet import clock

from pyglet.window import key
import torch
import mymap
from rocket import rocket
import ai
import population




window = pyglet.window.Window()
mymap.window_dimensions = (window.width, window.height)
batch = pyglet.graphics.Batch()

keys = key.KeyStateHandler()
window.push_handlers(keys)

mymap.initObstacle(batch)
mymap.init_reward_gates(batch)

fps_display = pyglet.window.FPSDisplay(window=window)



rocket = rocket(batch, None, False)



@window.event
def on_draw():
    window.clear()
    batch.draw()
    fps_display.draw()


mpopulation = population.Population(2, batch)



def update(dt):
    if(mpopulation.alive_agents > 0):
        #print("update")
        #mpopulation.update(dt)
        mpopulation.update_rockets()
        pyglet.clock.tick()
    # else:
    #     print("----------------------")
    #     print("new generation")
    #     if(not mpopulation.calculating):
    #         mpopulation.selection()
    
    if(rocket.status == "alive"):
        rocket.update(dt, keys)

    
    
    

clock.schedule_interval(update, 1/60.0 )


pyglet.app.run()

# dt = 0

# while dt<100:
#     dt = clock.tick()

# clock.unschedule(move, p_x)
