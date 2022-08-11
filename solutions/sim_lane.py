
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

env = DuckietownEnv(
	seed=8,
	map_name="small_loop",
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
gradient = abs(X - center_x) + abs(Y - h)
gradient = gradient.astype(np.float32)

# Scale gradient to be between 0-1
maxNum = np.max(gradient)
minNum = np.min(gradient)
for i in range(gradient.shape[0]):
	for j in range(gradient.shape[1]):
		gradient[i][j] = rescale(gradient[i][j], minNum, maxNum, 1, 0)
gradient[:gradient.shape[0] // 3,:] = 0
		
def update(dt):
	obs = env.render_obs()

	action = np.array([0.22, 0.0])
	
	# Convert img to grayscale and blur it
	img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
	img = cv2.GaussianBlur(img, (0,0), 1)
	
	# Compute sobel x & y
	sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
	
	# Compute positive and negative sobel gradients
	mask_sobelx_pos = (sobelx > 0)
	mask_sobelx_neg = (sobelx < 0)
	mask_sobely_pos = (sobely > 0)
	mask_sobely_neg = (sobely < 0)

	# Compute total gradient magnitude and threshold it
	Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
	mask_mag = (Gmag > 50)
	
	# Convert img to hsv for color masking
	imghsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
	white_lower_hsv = np.array([0, 0, 100])
	white_upper_hsv = np.array([255, 75, 255])
	yellow_lower_hsv = np.array([20, 50, 50])
	yellow_upper_hsv = np.array([40, 255, 255])

	# Mask white and yellow colors
	mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
	mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
	
	# Combine all masks
	mask_left_edge = mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
	mask_right_edge = mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
	
	# Mask gradient
	
	l_gradient = gradient.copy()
	l_gradient[:, gradient.shape[1] // 2:] = 1
	l_gradient[:l_gradient.shape[0] // 3,:] = 0
	r_gradient = gradient.copy()
	r_gradient[:, :gradient.shape[1] // 2] = 1
	r_gradient[:r_gradient.shape[0] // 3,:] = 0

	# Compute total of masks
	left_tot = np.sum(mask_left_edge * l_gradient)
	right_tot = np.sum(mask_right_edge * r_gradient)
	
	maxNum = mask_left_edge.shape[0] * (mask_left_edge.shape[1] // 2)
	left_tot = rescale(left_tot, 0, maxNum, 0, 1)
	right_tot = rescale(right_tot, 0, maxNum, 0, 1)
	
	action[1] = right_tot - left_tot
	print(action)
	# Speed boost
	if key_handler[key.LSHIFT]:
		action *= 1.5

	obs, reward, done, info = env.step(action)

	if key_handler[key.RETURN]:
		im = Image.fromarray(obs)
		im.save("screen.png")

	if done:
		env.close()
		sys.exit(0)

	env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
