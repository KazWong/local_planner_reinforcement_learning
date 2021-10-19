import numpy as np
import random
import os
import sys
import signal
import argparse
from argparse import Namespace

from omni.isaac.python_app import OmniKitHelper

class IsaacEnv(Env):
	def __init__(self, cfg_names):
        self.cfg_names = cfg_names

    def get_states(self):
        pass

    def get_rewards(self):
        pass

    def reset(self):
        pass

    def step_discrete(self, action):
        pass

    def robot_control(self, action):
        pass
