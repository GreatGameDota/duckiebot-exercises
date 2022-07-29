
# https://github.com/duckietown/gym-duckietown

# pip install duckietown-gym-daffy==6.0.1

from PIL import Image
import sys
import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# Modified from: https://github.com/duckietown/gym-duckietown/blob/daffy/manual_control.py

env = DuckietownEnv(
    seed=1,
    map_name="loop_pedestrians",
    domain_rand=False,
    frame_skip=1
)

obs = env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):    
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    
    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2
    
    """
    First value is speed of both wheels
    Second value is left wheel if positive and right wheel when negative
    """
    # action = np.array([0.0, -0.2])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5
        
    obs, reward, done, info = env.step(action)
    # obs is the returned image from the camera
    seg = env.render_obs(True)
    
    
    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)
        im.save("screen.png")
        
        seg = Image.fromarray(seg)
        seg.save("screen2.png")
        
    if done:
        print("done!")
        env.reset()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
