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
mymap.update_current_stage()

fps_display = pyglet.window.FPSDisplay(window=window)
generation = 0
generationLabel = pyglet.text.Label(('Generation: ' + str(generation)),
                        font_name='Times New Roman',
                        font_size=14,
                        x=60, y=460,
                        anchor_x='center', anchor_y='center')





@window.event
def on_draw():
    window.clear()
    batch.draw()
    generationLabel.draw()
    fps_display.draw()

#rocket = rocket(batch, None, False)
mpopulation = population.Population(200, batch)



def update(dt):
    #print(mpopulation.alive_agents)
    global generation
    if(mpopulation.alive_agents > 0):
        #print("update")
        mpopulation.update_rockets(keys)
    
    else:
        print("----------------------")
        print("new generation")
        generation+=1
        generationLabel.text = ('Generation: ' + str(generation))


        if(not mpopulation.calculating):
            mpopulation.calculate_shortest_distance_to_reward()
            newPop = mpopulation.selection()

            if((mymap.current_stage > 0 and generation == 10) or generation == 30):
                generation = 0
                generationLabel.text = ('Generation: ' + str(generation))
                mymap.current_stage +=1
                mymap.update_current_stage()
            mpopulation.reset(newPop)
    pyglet.clock.tick()
    # if(rocket.status == "alive"):
    #     rocket.update(dt, keys)

    
    
    

clock.schedule_interval(update, 1/60.0 )


pyglet.app.run()

# dt = 0

# while dt<100:
#     dt = clock.tick()

# clock.unschedule(move, p_x)
