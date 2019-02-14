import numpy as np

import tensorflow as tf

from .abstract_agent import AbstractAgent


class DQNAgent(AbstractAgent):

    """
    Description:
        A deep Q-learning agent, who tries to maximize to total accumulated
        discounted reward over all future time steps given the current state.
    """

    n_hidden = 512
    learning_rate = 0.001
    momentum = 0.95
    replay_memory_size = 500000
    eps_min = 0.1
    eps_max = 1.0
    eps_decay_steps = 2000000
    training_interval = 4  # Only train every X iterations
    save_steps = 1000  # Save the model every X training steps
    copy_steps = 10000  # Copy online DQN to target DQN every X training steps
    discount_rate = 0.99
    skip_start = 90  # Skip the start of every game (it's just waiting time).
    batch_size = 50
    checkpoint_path = "./my_dqn.ckpt"

    def q_network(self, inputs, name):
        """
        Creates a q_network with 'inputs' many inputs.
        """

        initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(name) as scope:
            hidden = tf.layers.dense(inputs, n_hidden,
                                     activation=tf.nn.relu, # hidden activation
                                     kernel_initializer=initializer)
            outputs = tf.layers.dense(hidden, self.n_outputs,
                                      kernel_initializer=initializer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        return outputs, trainable_vars_by_name

    def sample_memories(self):
        """
        Samples memories from experience replay buffer.
        """

        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, cont.
        for i in indices:
            memory = replay_memory[i]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
                cols[4].reshape(-1, 1))


    def __init__(self, n_inputs, n_outputs):
        super().__init__(n_inputs, n_outputs)
                                            # shape=[1:st dim,    h,   w, chs]
        self.state = tf.placeholder(tf.float32, shape=[None, n_inputs, 1, 1])
        self.online_qs, self.online_vars = q_network(self.state,
                                                     name='q_network/online')
        self.target_qs, self.target_vars = q_network(self.state,
                                                     name='q_network/target')

        copy_ops = [target_var.assign(self.online_vars[var_name])
                    for var_name, target_var in self.target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_ops)

        # Placeholder used for training.
        # Get Q-value prediction for the taken action only.
        self.x_action = tf.placeholder(tf.int32, shape=[None])
        q_value = tf.reduce_sum(target_qs * tf.one_hot(self.x_action,
                                                       self.n_outputs),
                                axis=1, keep_dims=True)

        # Placeholder y for providing target Q-values. Square errors < 1.0.
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        error = tf.abs(self.y - q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        # Create Nesterov Accelerated Gradient optimizer for minimizing loss
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        self.training_op = optimizer.minimize(loss, global_step=global_step)

        # Init tensorflow
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        self.replay_memory = deque([], maxlen=replay_memory_size)

        self.step = 0
        self.iteration = 0


    def get_action(self, state):
        self.step = self.global_step.eval()
        self.iteration += 1

        # P(Random action)=epsilon, else let online DQN evaluate what to do
        epsilon = max(eps_min, eps_max -(eps_max-eps_min)*step/eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            q_values = self.online_qs.eval(feed_dict={self.state: [state]})
            return np.argmax(q_values)  # optimal action

    def observe(self, prev_state, next_state, action, reward, done):
        # Save the experience in memory buffer
        self.replay_memory.append((prev_state, action, reward, next_state,
                                    1.0 - done))

        if self.iteration % training_interval != 0: # Don't overexert the agent
            return

        # Sample memories and use the target DQN to produce the target Q-Val
        x_state_val, x_action_val, rewards, x_next_state_val, continues = (
            sample_memories())
        next_q_values = self.target_qs.eval(
            feed_dict={self.state: x_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Training the online DQN
        self.training_op.run(feed_dict={self.state: x_state_val,
                                        self.x_action: x_action_val,
                                        self.y: y_val})

        with tf.Session() as sess:
            # Regularly copy the online DQN to the target DQN
            if self.step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            #if self.step % save_steps == 0:
                #saver.save(sess, checkpoint_path)


