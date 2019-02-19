import numpy as np
from collections import deque
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense

from .abstract_agent import AbstractAgent


class DQNAgent(AbstractAgent):

    """
    Description:
        A deep Q-learning agent, who tries to maximize to total accumulated
        discounted reward over all future time steps given the current state.
    """

    #n_hidden = 512
    n_hidden = 64
    hidden_layers = 2
    #learning_rate = 0.05
    learning_rate = 0.0005
    #replay_memory_size = 500000
    replay_memory_size = 25000
    training_interval = 1  # Only train every X iterations
    #eps_min = 0.1
    eps_min = 0.02
    eps_max = 1.0
    #eps_decay_steps = 150000
    eps_decay_steps = 75000
    #copy_steps = 10000  # Copy online DQN to target DQN every X training steps
    copy_steps = 5000  # Copy online DQN to target DQN every X training steps
    discount_rate = 0.99
    batch_size = 64
    activation = 'tanh' #'relu'
    optimizer_name = None


    def __init__(self, n_inputs, n_outputs):
        super().__init__(n_inputs, n_outputs)
        self.state = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.online_qs, self.online_vars = self.q_network(n_inputs)
        self.target_qs, self.target_vars = self.q_network(n_inputs)

        online_weights = self.online_vars.trainable_weights
        target_weights = self.target_vars.trainable_weights
        copy_ops = [target_weights[i].assign(online_weights[i])
                    for i in range(len(target_weights))]
        self.copy_online_to_target = tf.group(*copy_ops)

        # Placeholder used for training.
        # Get Q-value prediction for the taken action only.
        self.x_action = tf.placeholder(tf.int32, shape=[None])
        q_value = tf.reduce_sum(self.target_qs * tf.one_hot(self.x_action,
                                                       self.n_outputs),
                                axis=1, keepdims=True)

        # Placeholder y for providing target Q-values. Square errors < 1.0.
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        error = tf.abs(self.y - q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        #linear_error = 2 * (error - clipped_error)
        linear_error = error - clipped_error
        #loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
        loss = tf.reduce_mean(0.5 * tf.square(clipped_error) + linear_error)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=DQNAgent.learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer()#learning_rate=DQNAgent.learning_rate)
        self.optimizer_name = optimizer.get_name()
        self.training_op = optimizer.minimize(loss,
                                              global_step=self.global_step)

        # Init tensorflow
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        self.copy_online_to_target.run(session=self.sess)
        self.saver = tf.train.Saver()

        self.replay_memory = deque([], maxlen=DQNAgent.replay_memory_size)

        self.step = 0
        self.iteration = 0


    def q_network(self, inputs):
        """
        Creates a q_network with 'inputs' many inputs.
        """
        model = tf.keras.models.Sequential()

        model.add(Dense(DQNAgent.n_hidden,
                        input_shape=(inputs,),
                        activation=DQNAgent.activation))
        for i in range(DQNAgent.hidden_layers - 1):
            model.add(Dense(DQNAgent.n_hidden,
                            activation=DQNAgent.activation))
        model.add(Dense(self.n_outputs))

        q_values = model(self.state)

        return q_values, model

    def sample_memories(self):
        """
        Samples memories from experience replay buffer.
        """

        indices = np.random.permutation(len(self.replay_memory))[:DQNAgent.batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, cont.
        for i in indices:
            memory = self.replay_memory[i]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
                cols[4].reshape(-1, 1))

    def get_eps(self):
        eps_min = DQNAgent.eps_min; eps_max = DQNAgent.eps_max
        return max(eps_min, eps_max
                   - (eps_max-eps_min)*self.step/DQNAgent.eps_decay_steps)

    def get_action(self, state):
        self.step = self.global_step.eval(session=self.sess)
        self.iteration += 1

        # P(Random action)=epsilon, else let online DQN evaluate what to do
        eps_min = DQNAgent.eps_min; eps_max = DQNAgent.eps_max
        epsilon = self.get_eps()
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            q_values = self.online_qs.eval(feed_dict={self.state: [state]},
                                           session=self.sess)
            return np.argmax(q_values)  # optimal action

    def observe(self, prev_state, next_state, action, reward, done):
        # Save the experience in memory buffer
        self.replay_memory.append((prev_state, action, reward, next_state,
                                    1.0 - done))  # 1-done to cancel on death

        if self.iteration % DQNAgent.training_interval != 0: # Don't overexert
            return

        # Sample memories and use the target DQN to produce the target Q-Val
        x_state_val, x_action_val, rewards, x_next_state_val, continues = (
            self.sample_memories())
        next_q_values = self.target_qs.eval(
            feed_dict={self.state: x_state_val}, session=self.sess)
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * DQNAgent.discount_rate*max_next_q_values

        # Training the online DQN
        self.training_op.run(feed_dict={self.state: x_state_val,
                                        self.x_action: x_action_val,
                                        self.y: y_val},
                             session=self.sess)

        # Regularly copy the online DQN to the target DQN
        if self.step % DQNAgent.copy_steps == 0:
            self.copy_online_to_target.run(session=self.sess)

    def save_agent(self, file_path):
        # ...save regularly
        self.saver.save(self.sess, file_path)
        print('Agent saved to ' + file_path + ' after taking {} steps.'.format(
            self.step))

    def load_agent(self, file_path):
        if os.path.isfile(file_path + '.index'):
            self.saver.restore(self.sess, file_path)
            print('Agent loaded from ' + file_path)
        else:
            print('No agent loaded.')

