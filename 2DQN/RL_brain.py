import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(self,
                 n_actions,  # 输出节点数：神经网络输出多少个Q-value的值
                 n_features,  # 输入特征数：接收多少个observation
                 learning_rate=0.005,
                 reward_decay=0.9,  # gamma
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=3000,
                 batch_size=32,
                 e_greedy_increment=None,  # 不断的缩小随机的范围
                 output_graph=False,
                 double_q=True,
                 sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # 90%选择分值最大的行为
        self.replace_target_iter = replace_target_iter  # 每隔多少步更换一次target网络
        self.memory_size = memory_size  # 记录多少条transition的数据
        self.batch_size = batch_size  # mini-batch GD
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q

        # 总的迭代次数
        self.learn_step_counter = 0

        # 初始化记忆 [s, a, r, s_]
        # 因为观测s是输入特征，s_也是观测特征，这俩就是 n_features * 2
        # action行为是一种行为对应的索引号值，reward也是一个值，这俩就是 +2
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 包含 [target_net, evaluate_net]
        self._build_net()

        # 把最新的参数放到Q现实的神经网络当中去
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # 把eval网络的参数赋给target网络的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # build evaluate net
        # 为了计算损失有得到预测的输入state，还有真实的输出q_target
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')
        self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            # 配置层里面参数等
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # 第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.nn.relu(tf.matmul(l1, w2) + b2)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))
        with tf.name_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # build target net
        # 在Q现实的时候输入的是下一个观测s值_
        self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.nn.relu(tf.matmul(l1, w2) + b2)

    def store_transition(self, s, a, r, s_):
        # 如何去存储记忆
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 一开始加一个维度，对应是batch那个维度
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # 90% 概率选择最大的值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            # 10% 随机选择
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 首先判断要不要把eval网络参数赋给target网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从记忆库中采样一个批次的记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 根据下一时刻j+1时刻的观测通过target经验网络得到j+1时候的现实q_next（每种行为该给多少分）
        # 根据下一时刻j+1时刻的观测通过eval现有网络模型估计j+1时刻的预测q_eval4next（每种行为该给多少分）
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation

        # 特别注意：Q-Learn就是关于j+1时刻的现实通过衰减反推（取最大）得到j时刻的现实，再与j时刻的预测去计算误差（梯度）去调参的
        # 这里依然是Q-Learn的思想，所以首先会求出来j+1时刻的现实q_next，再取j+1时刻的现实通过衰减反推（取最大）得到j时刻的现实

        # 还需注意：下一行之所以运行的是eval网络而不是target网络，因为当前需要调的就是eval网络，所以回头Loss中只有关于eval的参数
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        # 重新改名叫q_target是因为经过取最大并赋值后，它才是Loss中要的j时刻的目标变量
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 获取每行索引号为n_feartures的对应的action行为的索引号
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            # 用真正会update参数的network来选action，用不动的network去算value
            max_act4next = np.argmax(q_eval4next,
                                     axis=1)  # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            # 用不动的network去算value
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        # 仅将可以使得j+1时刻收益最大的j时刻的action对应的分值进行反推，并应用Q-Learn反推逻辑赋值修改
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # 这里sess.run()会重新正向传播得到q_eval，计算重复~ 如果更好可以优化代码减少这里的重复计算
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # 增加epsilon，学的越来越好慢慢随机的可能性减小
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()























