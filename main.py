import pyglet
from pyglet.gl import *

from pyglet import clock

from pyglet.window import key
from map import initObstacle
from rocket import rocket



window = pyglet.window.Window()
batch = pyglet.graphics.Batch()

keys = key.KeyStateHandler()
window.push_handlers(keys)

initObstacle(batch)





rocket = rocket(batch)



@window.event
def on_draw():
    window.clear()
    batch.draw()

# def move(dt, p_x):
#    p_x += dt * 10




def update(dt):
    rocket.update(dt, keys)
    #rocket.check_collisions(obstacle)
    

clock.schedule_interval(update, 1/90.0 )

pyglet.app.run()

# dt = 0

# while dt<100:
#     dt = clock.tick()

# clock.unschedule(move, p_x)
