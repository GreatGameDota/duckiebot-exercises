
# https://github.com/duckietown/gym-duckietown

# pip install duckietown-gym-daffy==6.0.1

from PIL import Image
import sys
import os
import cv2
import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# Modified from: https://github.com/duckietown/gym-duckietown/blob/daffy/manual_control.py

os.system("mv brait_map.yaml /home/USER/.local/lib/python3.8/site-packages/gym_duckietown/maps/brait_map.yaml")

env = DuckietownEnv(
	seed=1,
	map_name="brait_map",
	domain_rand=False,
	frame_skip=1
)

env.reset()
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

# Rescale a number from its bounds to a different bound #
def rescale(n, lower1, upper1, lower2, upper2):
	return (n - lower1) / (upper1 - lower1) * (upper2 - lower2) + lower2

# Compute gradient #
h = 480
w = 640
Y, X = np.ogrid[:h, :w]
# Experiment with this!
center_x = w // 2
center_y = h // 2
gradient = np.sqrt((X - center_x)**2 + (Y - center_y)**2) + 2*abs(X - center_x) + 3*abs(Y - h)

# Scale gradient to be between 0-1
maxNum = np.max(gradient)
minNum = np.min(gradient)
for i in range(gradient.shape[0]):
	for j in range(gradient.shape[1]):
		gradient[i][j] = rescale(gradient[i][j], minNum, maxNum, 1, 0)
		
def update(dt):
	obs = env.render_obs()

	action = np.array([0.22, 0.0])

	# Convert to hsv for color masking
	hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

	# CHANGE ME
	lower_hsv = np.array([20, 250, 0])
	upper_hsv = np.array([40, 300, 255])

	color_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

	# Experiment with this! (Use the second colab notebook)
	left_mask = color_mask[:, :color_mask.shape[1] // 2]
	right_mask = color_mask[:, color_mask.shape[1] // 2:]
	
	l_gradient = gradient[:, :color_mask.shape[1] // 2]
	r_gradient = gradient[:, color_mask.shape[1] // 2:]

	left_tot = np.sum(left_mask * l_gradient)
	right_tot = np.sum(right_mask * r_gradient)
	
	maxNum = left_mask.shape[0] * left_mask.shape[1]
	left_tot = rescale(left_tot, 0, maxNum, 0, 1) + 0.01
	right_tot = rescale(right_tot, 0, maxNum, 0, 1)
	
	action[1] += right_tot - left_tot

	# Speed boost
	if key_handler[key.LSHIFT]:
		action *= 1.5

	obs, reward, done, info = env.step(action)

	if key_handler[key.RETURN]:
		im = Image.fromarray(obs)
		im.save("screen.png")

	if done:
		print("done!")
		env.reset()

	env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
