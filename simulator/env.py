

class Env(object):
    def __init__(self, cfg_names):
        self.cfg_names = cfg_names
        self.read_yaml(cfg_names[0])
        self.control_hz = 0.2
        self.collision_th = 0.5
        self.image_size = (60, 60)
        self.image_batch = 1
        self.done = 0
        self.epoch = 0
        self.reset_count = 0

        self.discrete_actions = []

    def set_img_size(self, img_size):
        self.image_size = img_size
        self.init()

    def set_colis_dist(self, dist):
        self.collision_th = dist

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
