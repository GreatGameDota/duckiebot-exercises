#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import cv2
from aido_schemas import (
	Context,
	DB20Commands,
	DB20Observations,
	EpisodeStart,
	GetCommands,
	JPGImage,
	LEDSCommands,
	protocol_agent_DB20,
	PWMCommands,
	RGB,
	wrap_direct,
)

class Agent:
	left: Optional[np.ndarray]
	right: Optional[np.ndarray]
	rgb: Optional[np.ndarray]

	def init(self, context: Context):
		context.info("init()")
		self.rgb = None
		self.start_time = None
		
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
				gradient[i][j] = self.rescale(gradient[i][j], minNum, maxNum, 1, 0)
		gradient[:gradient.shape[0] // 3,:] = 0
		self.gradient = gradient

	def on_received_seed(self, data: int):
		np.random.seed(data)

	def on_received_episode_start(self, context: Context, data: EpisodeStart):
		context.info(f'Starting episode "{data.episode_name}".')

	def on_received_observations(self, context: Context, data: DB20Observations):
		camera: JPGImage = data.camera
		if self.rgb is None:
			context.info("received first observations")
		self.rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(camera.jpg_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

	def compute_commands(self) -> Tuple[float, float]:
		""" Returns the commands (pwm_left, pwm_right) """
		# If we have not received any image, we don't move
		if self.rgb is None:
			return 0.0, 0.0
			
		obs = self.rgb
		
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
		
		l_gradient = self.gradient.copy()
		left_mask_coord = self.gradient.shape[1] - self.gradient.shape[1] // 6
		l_gradient[:, left_mask_coord:] = 0
		l_gradient[:, self.gradient.shape[1] // 2:left_mask_coord] = 0.5
		
		r_gradient = self.gradient.copy()
		r_gradient[:, :self.gradient.shape[1] // 4] = 0
		r_gradient[:, self.gradient.shape[1] // 4:self.gradient.shape[1] // 2] = 0.5
		
		l_gradient[:l_gradient.shape[0] // 3,:] = 0
		r_gradient[:r_gradient.shape[0] // 3,:] = 0

		# Compute total of masks
		left_tot = np.sum(mask_left_edge * l_gradient)
		right_tot = np.sum(mask_right_edge * r_gradient)
		
		# The 5 is randomly picked, maybe try experimenting with it!
		maxNum = mask_left_edge.shape[0] * (mask_left_edge.shape[1] // 2) * 5
		left_tot = self.rescale(left_tot, 0, maxNum, 0, 1)
		right_tot = self.rescale(right_tot, 0, maxNum, 0, 1)
		
		pwm_left = 0.15
		pwm_right = 0.15
		
		tot = left_tot - right_tot
		if tot > 0:
			pwm_left += self.trim(tot, 0, 0.85)
		else:
			pwm_right += self.trim(-tot, 0, 0.85)
		return pwm_left, pwm_right

	def on_received_get_commands(self, context: Context, data: GetCommands):
		pwm_left, pwm_right = self.compute_commands()

		stop_time = 60 * 5
		if self.start_time is None:
			self.start_time = data.at_time
		elif data.at_time - self.start_time > stop_time:
			pwm_left = 0.0
			pwm_right = 0.0
		if data.at_time - self.start_time > stop_time + 5:
			context.info("finish()")
			exit(0)
			
		col = RGB(0.0, 0.0, 1.0)
		col_left = RGB(pwm_left, pwm_left, 0.0)
		col_right = RGB(pwm_right, pwm_right, 0.0)
		led_commands = LEDSCommands(col, col_left, col_right, col_left, col_right)
		pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
		commands = DB20Commands(pwm_commands, led_commands)
		context.write("commands", commands)
		
	def finish(self, context: Context):
		context.info("finish()")
		
	def rescale(self, n, lower1, upper1, lower2, upper2):
		"""
		Rescale a number from its bounds to a different bound
		"""
		return (n - lower1) / (upper1 - lower1) * (upper2 - lower2) + lower2
		
	def trim(self, value, low, high):
		"""
		Trims a value to be between some bounds.

		Args:
			value: the value to be trimmed
			low: the minimum bound
			high: the maximum bound

		Returns:
			the trimmed value
		"""
		return max(min(value, high), low)


def main():
	node = Agent()
	protocol = protocol_agent_DB20
	wrap_direct(node=node, protocol=protocol)

if __name__ == "__main__":
	main()

