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
		gradient = np.sqrt((X - center_x)**2 + (Y - center_y)**2) + 2*abs(X - center_x) + 3*abs(Y - h)

		# Scale gradient to be between 0-1
		maxNum = np.max(gradient)
		minNum = np.min(gradient)
		for i in range(gradient.shape[0]):
			for j in range(gradient.shape[1]):
				gradient[i][j] = self.rescale(gradient[i][j], minNum, maxNum, 1, 0)
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
		
		# Convert to hsv for color masking
		hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

		# CHANGE ME
		lower_hsv = np.array([0, 100, 0])
		upper_hsv = np.array([40, 255, 255])

		color_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
		color_mask[:color_mask.shape[0]//3,:] = 0

		# Experiment with this! (Use the second colab notebook)
		left_mask = color_mask[:, :color_mask.shape[1] // 2]
		right_mask = color_mask[:, color_mask.shape[1] // 2:]
		
		l_gradient = self.gradient[:, :color_mask.shape[1] // 2]
		r_gradient = self.gradient[:, color_mask.shape[1] // 2:]

		left_tot = np.sum(left_mask * l_gradient)
		right_tot = np.sum(right_mask * r_gradient)
		
		maxNum = left_mask.shape[0] * left_mask.shape[1] * 7
		left_tot = self.rescale(left_tot, 0, maxNum, 0, 1) + 0.01
		right_tot = self.rescale(right_tot, 0, maxNum, 0, 1)
		
		pwm_left = 0.05
		pwm_right = 0.05
		
		tot = left_tot - right_tot
		if tot > 0:
			pwm_left += self.trim(tot, 0, 0.85)
		else:
			pwm_right += self.trim(-tot, 0, 0.85)
		return pwm_left, pwm_right

	def on_received_get_commands(self, context: Context, data: GetCommands):
		pwm_left, pwm_right = self.compute_commands()

		stop_time = 60 * 0.5
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

