import sys
sys.path.append('../utils/')
import numpy as np
import time, atexit, os
import tensorflow.compat.v1 as tf
import tf_slim as slim
from logger import EpochLogger

tf.disable_v2_behavior()
tf.enable_resource_variables()

#for np.random.uniform
np.random.seed(1)
tf.set_random_seed(1)


def setup_logger_kwargs(exp_name='CurrentExp', output_fname='progress.txt', data_dir=None, datestamp=False):

    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    logger_kwargs = dict(output_dir=relpath, output_fname=output_fname)
    return logger_kwargs


class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)    #for priority
        self.data = np.zeros(capacity, dtype=object)    #for states
        self.data_pointer = 0

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1    #find the last row index
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  #replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        #propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  #stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01
    alpha = 0.6  #[0~1] convert the importance of TD error to priority
    beta = 0.4  #importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        #TODO: use the max_p to at least add some value
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       #priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  #avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            image_size=[64, 64, 3],
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,        #prioritized replay for different sample weight, random search will miss some less-frequent features
            restore_model=False,
            exp_name = None
    ):

        logger_kwargs = setup_logger_kwargs(exp_name)
        self.logger = EpochLogger(**logger_kwargs)
        self.n_actions = n_actions
        self.lr = learning_rate
        #gradual epsilon decays
        self.gamma = reward_decay
        #random factor to choose action
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.restore_model = restore_model
        self.n_features = self.image_size[0] * self.image_size[1] * self.image_size[2]
        self.dir_dim = 2
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        if self.restore_model:
            self.epsilon = self.epsilon_max

        self.prioritized = prioritized    #every sample in experience pool have a priority

        self.learn_step_counter = 0    #when to update target network

        self.build_net()

        self.replace_target_op = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(self.get_vars('eval_net'), self.get_vars('target_net'))])

        #data structure for storing state, action and reward, experience pool
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            #(S,S',A,R) for every memory
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)

        if self.restore_model == False:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.logger.restore_model(self.sess)

        #save pbtxt
        self.logger.save_model_info(self.sess)

        #only output the graph in tf events? may add more scalar, also save to output_dir
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        #TODO: no use
        self.cost_his = []

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def save_param(self):
        train_param = {}
        train_param['batch_size'] = self.batch_size
        train_param['memory_size'] = self.memory_size
        train_param['Learning_rate'] = self.lr
        train_param['e_greedy_increment'] = self.epsilon
        train_param['actions'] = self.n_actions
        train_param['image_size'] = self.image_size
        self.logger.save_config(train_param)

    #network to produce q-value
    def build_net(self, use_fullyconnected = True):
        def build_net(image, direction, trainable):
            with tf.variable_scope("image"):
                image_net = slim.conv2d(image, 32, [8, 8], stride=4, scope='conv1', trainable=trainable)
                image_net = slim.conv2d(image_net, 64, [4, 4], stride=2, scope='conv2', trainable=trainable)
                image_net = slim.conv2d(image_net, 64, [3, 3], stride=1, scope='conv3', trainable=trainable)
            with tf.variable_scope("direction"):
                dir_net = tf.layers.dense(direction, 64, activation=tf.nn.relu, trainable=trainable)
                dir_net = tf.reshape(dir_net, [-1,1,1,64])
                dir_net = tf.tile(dir_net, [1,8,8,1])
            with tf.variable_scope("connect"):
                net = tf.add(image_net, dir_net)
            with tf.variable_scope("convs"):
                net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv1', trainable=trainable)
                net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv2', trainable=trainable)
                net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv3', trainable=trainable)
                net = slim.flatten(net, scope='flatten')
            with tf.variable_scope("fc"):
                net = tf.layers.dense(net, units=512, activation=tf.nn.tanh, trainable=trainable)
                out = tf.layers.dense(net, units=512, activation=tf.nn.tanh, trainable=trainable)
                out = tf.layers.dense(out, self.n_actions, activation=None, trainable=trainable)
            return out

        #evaluate network
        self.image_input = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.image_size[2]], name='image_input')
        self.dir_input = tf.placeholder(tf.float32, [None, self.dir_dim], name='dir_input')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  #for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            self.q_eval = build_net(self.image_input,self.dir_input, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                #for updating Sumtree
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)
                #calculate the loss with sample probability
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        #target network not trainable, update from eval network, improve training stability
        self.image_input_ = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.image_size[2]], name='image_input_')
        self.dir_input_ = tf.placeholder(tf.float32, [None, self.dir_dim], name='dir_input_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_net(self.image_input_,self.dir_input_, False)

    #store experience to the DQN memory
    def store_transition(self, s, a, r, s_):
        #prioritized replay to save (S,A,R,S') to memory buffer
        if self.prioritized:
            transition = np.hstack((s[1].flatten(), s[0], [a, r], s_[0], s_[1].flatten()))
            self.memory.store(transition)    #high priority for newly arrived transition
        else:       #random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s[1].flatten(), s[0], [a, r], s_[0], s_[1].flatten()))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation, actions,is_test=False):
        image_input = observation[1]
        dir_input = observation[0]
        image_input = image_input[np.newaxis, :]
        dir_input = dir_input[np.newaxis, :]
        if is_test:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.image_input: image_input, self.dir_input: dir_input})
            action = np.argmax(actions_value)
            print("is_test    ", action)
            return action
        else:
            if np.random.uniform() < self.epsilon:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.image_input: image_input, self.dir_input: dir_input})
                action = np.argmax(actions_value)
                print('model action:',action)
            #random action to learn new experience
            else:
                action = np.random.randint(0, self.n_actions)
            return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        #select batch_size experience to learn, each experience has a weight
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
        #choose the max q in n_action as the q_value
        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.image_input_: batch_memory[:, -self.n_features:].reshape(self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                           self.dir_input_: batch_memory[:, -self.n_features-self.dir_dim:-self.n_features],
                           self.image_input: batch_memory[:, :self.n_features].reshape(self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                           self.dir_input: batch_memory[:, self.n_features:self.n_features+self.dir_dim]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features+self.dir_dim].astype(int)
        reward = batch_memory[:, self.n_features+self.dir_dim + 1]

        #use target network to update because eval network change frequently
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.image_input: batch_memory[:, :self.n_features].reshape(self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                    self.dir_input: batch_memory[:, self.n_features:self.n_features+self.dir_dim],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            #update higher priority for bigger TD-error
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.image_input: batch_memory[:, :self.n_features].reshape(self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                                                    self.dir_input: batch_memory[:, self.n_features:self.n_features+self.dir_dim],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return abs_errors, self.cost

    def save_train_model(self):
        self.logger.save_train_model(self.sess)
