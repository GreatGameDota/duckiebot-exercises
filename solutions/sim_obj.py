
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
    map_name="loop_pedestrians",
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
        
        for det in pred[0]:
            det = reversed(det)
            det = det.cpu().numpy()
            cv2.rectangle(obs, (int(det[3]), int(det[2])), (int(det[5]), int(det[4])), (255,0,0), 2)

        im = Image.fromarray(obs)
        im.save(f'{env.step_count:03d}.png')
        
    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)
        im.save("screen.png")
        
        seg = Image.fromarray(seg)
        seg.save("screen2.png")
        
    if done:
        print("done!")
        env.reset()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
