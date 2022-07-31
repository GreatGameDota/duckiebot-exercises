#!/usr/bin/env python3

import os
import random
from functools import reduce

import cv2
import numpy as np
## Important - don't remove these imports even though they seem unneeded
import pyglet
from pyglet.window import key

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, xminyminxmaxymax2xywfnormalized, run, \
    train_test_split

from setup import find_all_boxes_and_classes



class SkipException(Exception):
    pass

# Need to change this dataset directory if not running inside docker container... TODO fix
DATASET_DIR="../duckietown_dataset"

if not os.path.isdir(DATASET_DIR):
    os.mkdir(DATASET_DIR)

os.system(f'rm -r {DATASET_DIR}/train')
os.system(f'rm -r {DATASET_DIR}/val')
os.mkdir(f'{DATASET_DIR}/train')
os.mkdir(f'{DATASET_DIR}/val')
os.mkdir(f'{DATASET_DIR}/train/labels')
os.mkdir(f'{DATASET_DIR}/val/labels')
os.mkdir(f'{DATASET_DIR}/train/images')
os.mkdir(f'{DATASET_DIR}/val/images')
    
if not os.path.isdir(f'{DATASET_DIR}/images'):
    os.mkdir(f'{DATASET_DIR}/images')
    os.mkdir(f'{DATASET_DIR}/labels')
    
IMAGE_SIZE=416
SPLIT_PERCENTAGE=0.8


npz_index = 0

def save_npz(img, boxes, classes):
    global npz_index

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{DATASET_DIR}/images/{npz_index}.jpg", img)
    boxes = np.array([xminyminxmaxymax2xywfnormalized(box, IMAGE_SIZE) for box in boxes])
    with open(f"{DATASET_DIR}/labels/{npz_index}.txt", "w") as f:
        for i in range(len(boxes)):
            f.write(f"{classes[i]} "+" ".join(map(str,boxes[i]))+"\n")
            
    print(f"COLLECTED IMAGE #{npz_index}")

    npz_index += 1

# some setup
seed(1234)
MAX_STEPS = 1000
nb_of_steps = 0

# we interate over several maps to get more diverse data
possible_maps = [
    "loop_pedestrians",
    "udem1",
    "loop_dyn_duckiebots",
    "loop_obstacles",
]
env_id = random.randint(0, 3)
env = None
while True:
    if env is not None:
        try:
            env.window.close()
            env.close()
            env_id = random.randint(0, 3)
        except:
            pass
        
    if env_id >= len(possible_maps):
        env_id = env_id % len(possible_maps)
    env = launch_env(possible_maps[env_id])
    print(f'###### {env_id} ######')
    policy = PurePursuitPolicy(env)
    obs = env.reset()

    inner_steps = 0
    if nb_of_steps >= MAX_STEPS:
        break

    while True:
        if nb_of_steps >= MAX_STEPS or inner_steps > 100:
            break

        action = policy.predict(np.array(obs))

        obs, rew, done, misc = env.step(action)
        seg = env.render_obs(True)


        obs = cv2.resize(obs, (IMAGE_SIZE, IMAGE_SIZE))
        seg = cv2.resize(seg, (IMAGE_SIZE, IMAGE_SIZE))

        env.render()

        try:
            boxes, classes = find_all_boxes_and_classes(seg)
        except SkipException as e:
            print(e)
            continue

        # TODO uncomment me if you want to save images with bounding boxes (this will NOT work for training, but is useful for debugging)
        #for box in boxes:
            #pt1 = (box[0], box[1])
            #pt2 = (box[2], box[3])
            #cv2.rectangle(obs, pt1, pt2, (255,0,0), 2)

        save_npz(obs, boxes, classes)
        nb_of_steps += 1
        inner_steps += 1

        if done or inner_steps % 100 == 0:
            env.reset()
    if nb_of_steps >= MAX_STEPS:
        break

env.window.close()
env.close()
print("NOW GOING TO MOVE IMAGES INTO TRAIN AND VAL")
all_image_names = [str(idx) for idx in range(npz_index)]
train_test_split(all_image_names, SPLIT_PERCENTAGE, DATASET_DIR)
print("DONE!")
run(f"rm -rf {DATASET_DIR}/images {DATASET_DIR}/labels")

