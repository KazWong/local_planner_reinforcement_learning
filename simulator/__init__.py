
class Env(object):
    def __init__(self, cfg_names):

    def _get_states(self, save_img=None):

    def _get_rewards(self, state, min_dist, is_collision):

    def reset(self, target_dist = 0.0):

    def step(self, action):

    def step_discrete(self, action):

    def image_trans(self, img_ros):

    def state_callback(self, msg):

    def get_avoid_areas(self):

    def reset_robots(self, target_dist=0.0):

    def del_all_obs(self):

    def reset_obs(self):

    def get_model_state(self):

    def get_robots_state(self):

    def random_pose(self, x, y, sita):

    def free_check_robot(self, x, y, robot_poses):

    def free_check_obj(self, target_pose, obj_poses):

    def random_robots_pose(self, pose_ranges):

    def robot_control(self, action):

    def get_robot_name(self, i):

    def empty_robots(self):

    def set_img_size(self, img_size):

    def set_colis_dist(self, dist):

    def read_yaml(self, yaml_file):

    def init_datas(self):
