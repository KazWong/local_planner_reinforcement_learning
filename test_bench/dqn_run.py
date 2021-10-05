import sys
sys.path.append('../network/')
sys.path.append('../simulator/')
from network_rl import DQNPrioritizedReplay
import tensorflow as tf
import numpy as np
from gazebo_env import GazeboEnv as ENV
import rospy
import math

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

is_train = True
is_restore_model = False
#is_train = False
#is_restore_model = True


MEMORY_SIZE = 50000    #sumtree size
BATCH_SIZE = 1024
EXP_NAME = 'rl_lscm'
Learning_rate = 0.0005
E_greedy_increment = 0.00005    #random or network update, e-greedy algo

if is_train:
    env = ENV(['easy_difficulty_env.yaml'])
else:
    env = ENV(['easy_difficulty_env.yaml'])
RL_prio = DQNPrioritizedReplay(
        n_actions=len(env.discrete_actions), image_size=[env.image_size[0], env.image_size[1], env.image_batch],
        learning_rate=Learning_rate, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE,
        e_greedy_increment=E_greedy_increment, restore_model=is_restore_model, prioritized=True, output_graph=True,
        exp_name=EXP_NAME)


def train(RL):
    total_steps = 0
    epochs = 1000000
    steps_per_epoch = 1500
    episode_max_len = 300
    for epoch in range(epochs):
        #every epoch, save model and output summary log
        env.epoch = epoch
        observation = env.reset(target_dist = 0.5)
        #exceed this value means cannot reach goal in certain time
        ep_len = 0.0
        #reward accumulation
        ep_return = 0.0
        for step in range(steps_per_epoch):
            #choose action from eval net
            action = RL.choose_action([observation[0], observation[1]], env.discrete_actions, is_test=False)
            '''
            #choose action from dwa
            th = math.atan2(observation[0][1], observation[0][0])
            states = np.array([0, 0, 0, observation[5].linear.x, observation[5].linear.y, observation[5].angular.z])
            goal = np.array([observation[0][0], observation[0][1], th])
            print("states: ", states)
            print("goal:", goal)
            action = DWA.choose_action(states, goal, observation[4])
            '''
            observation_, reward, done = env.step_discrete(action)
            #rospy.sleep(rospy.Duration(10, 0))
            if env.done != -100:
                #(S,A,R,S')
                RL.store_transition([observation[0], observation[1]], action, reward, [observation_[0], observation_[1]])
                ep_return += reward
                ep_len += 1
                if env.done < 0:
                    env.done = -100
            if ep_len > episode_max_len:
                env.done = -100
            observation = observation_
            if env.done == -100:
                print("epoch: ", epoch)
                print("ep_return: ", ep_return)
                print("ep_len", ep_len)
                env.robot_control([0,0])
                #
                total_steps += np.array(ep_len).sum()
                if total_steps > BATCH_SIZE:
                    for i in range(100):
                        #TODO: when to learn -> training performance
                        abs_errors, loss = RL.learn()
                    RL.save_train_model()
                ep_len = 0.0
                ep_return = 0.0
                observation = env.reset(target_dist = 0.5)
        #TODO: save ep_len, ep_return and loss
        if total_steps > BATCH_SIZE:
            RL.logger.log_tabular('epoch', epoch)
            RL.logger.log_tabular('step', total_steps)
            test(RL,20)


def test(RL,test_replay_):
    dt = env.control_hz
    env.set_colis_dist(0.5)
    #how much times to test
    test_replay = test_replay_
    #TODO: plot the performance
    collision = 0.0
    stuck = 0.0
    reach = 0.0
    av_reward = 0.0
    av_r_obs = 0.0
    step_per_collision = 0.0
    step_per_reach = 0.0
    av_vmax = 0.0
    av_v = 0.0
    av_trajectory_length = 0.0
    av_total_delta_v = 0.0
    av_wmax = 0.0
    av_w = 0.0
    av_total_theta = 0.0
    av_total_delta_w = 0.0

    for i in range(test_replay):
        observation = env.reset(target_dist = 0.5)
        steps = 0
        r_obs = 0.0
        is_end = 0
        ep_len = 0.0
        ep_return = 0.0
        #choose a zero velocity to start
        last_action = 3

        #motion analysis
        vmax = 0.0
        trajectory_length = 0.0
        total_delta_v = 0.0
        wmax = 0.0
        total_theta = 0.0
        total_delta_w = 0.0

        while True:
            steps += 1
            action = RL.choose_action([observation[0], observation[1]], env.discrete_actions, is_test=True)
            v = env.discrete_actions[action][0]
            w = env.discrete_actions[action][1]
            if v > vmax:
                vmax = v
            if math.fabs(w) > wmax:
                wmax = math.fabs(w)
            trajectory_length += dt * v
            total_theta += dt * math.fabs(w)
            total_delta_v += math.fabs(v - env.discrete_actions[last_action][0])
            total_delta_w += math.fabs(w - env.discrete_actions[last_action][1])

            observation_, reward, done = env.step_discrete(action)

            if is_end == 0:
                ep_return += reward
                ep_len += 1
                r_obs += dt * 1.0 / observation_[2]

                if env.done < 0 or steps > 200:
                    is_end = 1
                    av_reward += ep_return
                    av_vmax += vmax
                    av_wmax += wmax
                    if env.done == -1:
                        collision += 1
                        step_per_collision += ep_len
                    elif env.done == -2:
                        reach += 1
                        step_per_reach += ep_len
                        av_r_obs += r_obs
                        av_trajectory_length += trajectory_length
                        av_total_theta += total_theta
                        av_total_delta_v += total_delta_v
                        av_total_delta_w += total_delta_w
                    elif steps > 200:
                        stuck += 1

            print("done: ", env.done)
            if is_end == 1 or steps > 200:
                env.robot_control([0,0])
                break
            observation = observation_
            last_action = action
    if reach != 0:
        step_per_reach = step_per_reach / reach
        av_r_obs = av_r_obs / reach
        av_trajectory_length = av_trajectory_length / reach
        av_total_delta_v = av_total_delta_v / reach
        av_total_theta = av_total_theta / reach
        av_total_delta_w = av_total_delta_w / reach
        av_v = av_trajectory_length / (step_per_reach * dt)
        av_w = av_total_theta / (step_per_reach * dt)
    if collision != 0:
        step_per_collision = step_per_collision / collision
    av_reward = av_reward / test_replay
    av_vmax = av_vmax / test_replay
    av_wmax = av_wmax / test_replay

    RL.logger.log_tabular('replay_times', test_replay)
    RL.logger.log_tabular('reach_rate', reach / test_replay)
    RL.logger.log_tabular('reach_av_step', step_per_reach)
    RL.logger.log_tabular('collision_rate', collision / test_replay)
    RL.logger.log_tabular('collision_av_step', step_per_collision)
    RL.logger.log_tabular('stuck_rate', stuck / test_replay)
    RL.logger.log_tabular('average_Reward', av_reward)
    RL.logger.log_tabular('average_R_obs', av_r_obs)
    RL.logger.log_tabular('average_vmax', av_vmax)
    RL.logger.log_tabular('average_trajectory_length', av_trajectory_length)
    RL.logger.log_tabular('average_v', av_v)
    RL.logger.log_tabular('average_total_delta_v', av_total_delta_v)
    RL.logger.log_tabular('average_wmax', av_wmax)
    RL.logger.log_tabular('average_total_theta', av_total_theta)
    RL.logger.log_tabular('average_w', av_w)
    RL.logger.log_tabular('average_total_delta_w', av_total_delta_w)
    RL.logger.dump_tabular()


if is_train:
    train(RL_prio)
else:
    test(RL_prio, test_replay_=200)

