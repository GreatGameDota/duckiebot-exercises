
# https://github.com/duckietown/gym-duckietown
# pip install duckietown-gym-daffy==6.0.1

# git clone https://github.com/ultralytics/yolov5.git
# pip3 install -r requirements.txt

import torch
from PIL import Image
import os
import sys
import gym
import numpy as np
import cv2

import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

#### MODEL SETUP ####
os.system("touch data/duckietown.yaml")
with open('data/duckietown.yaml', 'w') as f:
	f.write("train: ./\n")
	f.write("val: ./\n")
	f.write("nc: 4\n")
	f.write("names: [ 'duckie', 'cone', 'truck', 'bus' ]")

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

device = select_device('cpu')
model = DetectMultiBackend("runs/train/exp/weights/best.onnx", device=device, data="data/duckietown.yaml", dnn=False, fp16=True)
model.warmup(imgsz=(1 if model.pt else 1, 3, 416, 416))
########

# Modified from: https://github.com/duckietown/gym-duckietown/blob/daffy/manual_control.py

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
	
	if env.step_count % 25 == 0:
		img = letterbox(obs, 416, model.stride, model.pt)[0]
		img = img.transpose((2, 0, 1))  # HWC to CHW
		img = np.ascontiguousarray(img)

		img = torch.from_numpy(img).to(device)
		img = img.half()
		# img = img.float()
		img /= 255
		img = img[None]

		pred = model(img, augment=False, visualize=False)
		pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)
		pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], obs.shape).round()

		mask = np.zeros((obs.shape[0], obs.shape[1]))
		for det in pred[0]:
			det = reversed(det)
			det = det.cpu().numpy()
			mask[int(det[4]):int(det[2]),int(det[5]):int(det[3])] = 500
		
		left_mask = mask[:, :mask.shape[1] // 2]
		right_mask = mask[:, mask.shape[1] // 2:]
		
		l_gradient = gradient[:, :mask.shape[1] // 2]
		r_gradient = gradient[:, mask.shape[1] // 2:]

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

pyglet.clock.schedule_interval(update, 1.0 / env.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
