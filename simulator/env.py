

class Env(object):
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
