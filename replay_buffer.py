import math
import numpy as np

class N_Steps_ReplayBuffer(object):  # without store gamma for multi step DQN、DynDQN_n_E、DynDQN_n_T、MMDQN
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.buffer_capacity = int(args.buffer_capacity)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.count = 0
        self.current_size = 0

        self.state_buf = np.zeros([self.buffer_capacity, self.state_dim])
        self.action_buf = np.zeros([self.buffer_capacity, 1])
        self.reward_buf = np.zeros(self.buffer_capacity)
        self.next_state_buf = np.zeros([self.buffer_capacity, self.state_dim])
        self.terminal_buf = np.zeros(self.buffer_capacity)
        self.done_buf = np.zeros(self.buffer_capacity)

    def store_transition(self, state, action, reward, next_state, terminal, done):
        self.state_buf[self.count] = state
        self.action_buf[self.count] = action
        self.reward_buf[self.count] = reward
        self.next_state_buf[self.count] = next_state
        self.terminal_buf[self.count] = terminal
        self.done_buf[self.count] = done

        self.count = (self.count + 1) % self.buffer_capacity
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample_indices(self, n_steps):
        return np.random.randint(0, self.current_size - (n_steps - 1), size=self.batch_size)

    def sample(self, n_steps, index):
        batch_state = np.zeros([self.batch_size, n_steps, self.state_dim])
        batch_action = np.zeros([self.batch_size, n_steps, 1])
        batch_reward = np.zeros([self.batch_size, n_steps])
        batch_next_state = np.zeros([self.batch_size, n_steps, self.state_dim])
        batch_terminal = np.zeros([self.batch_size, n_steps])
        batch_done = np.zeros([self.batch_size, n_steps])

        for i in range(n_steps):
            batch_state[:, i, :] = self.state_buf[index + i]
            batch_action[:, i, :] = self.action_buf[index + i]
            batch_reward[:, i] = self.reward_buf[index + i]
            batch_next_state[:, i, :] = self.next_state_buf[index + i]
            batch_terminal[:, i] = self.terminal_buf[index + i]
            batch_done[:, i] = self.done_buf[index + i]

        n_step_returns = np.zeros(self.batch_size)
        gammas_at_done = np.ones(self.batch_size) * self.gamma ** n_steps

        final_next_state = batch_next_state[:, -1, :]
        final_done = batch_terminal[:, -1]

        for j in range(self.batch_size):
            for i in range(n_steps):
                n_step_returns[j] += batch_reward[j, i] * math.pow(self.gamma, i)

                if batch_done[j, i]:
                    gammas_at_done[j] = math.pow(self.gamma, (i+1))
                    final_next_state[j, :] = batch_next_state[j, i, :]
                    final_done[j] = batch_terminal[j, i]

                    break

        return {
            'state': batch_state[:, 0, :],
            'action': batch_action[:, 0, :],
            'reward': n_step_returns,
            'next_state': final_next_state,
            'done': final_done,
            'gamma': gammas_at_done
        }

    def get_size(self):
        return self.current_size
    
class N_Steps_ReplayBuffer_(object):  # has store gamma for DQN_LNSS、ESDQN
	def __init__(self, args):
		self.state_dim = args.state_dim
		self.buffer_capacity = int(args.replay_buffer_capacity)
		self.batch_size = args.batch_size
		self.count = 0
		self.current_size = 0

		self.state_buf = np.zeros([self.buffer_capacity, self.state_dim])
		self.action_buf = np.zeros([self.buffer_capacity, 1])
		self.reward_buf = np.zeros(self.buffer_capacity)
		self.next_state_buf = np.zeros([self.buffer_capacity, self.state_dim])
		self.gamma_buf = np.zeros(self.buffer_capacity)
		self.done_buf = np.zeros(self.buffer_capacity)      

	def store_transition(self, state, action, reward, next_state, gamma, terminal, done):
		if done:
			done = terminal

		self.state_buf[self.count] = state
		self.action_buf[self.count] = action
		self.reward_buf[self.count] = reward
		self.next_state_buf[self.count] = next_state
		self.gamma_buf[self.count] = gamma
		self.done_buf[self.count] = done

		self.count = (self.count + 1) % self.buffer_capacity
		self.current_size = min(self.current_size + 1, self.buffer_capacity)

	def sample_indices(self, n_steps):
		return np.random.randint(0, self.current_size - (n_steps - 1), size=self.batch_size)

	def sample(self, n_steps, index):
		batch_state = np.zeros([self.batch_size, self.state_dim])
		batch_action = np.zeros([self.batch_size, 1])
		batch_reward = np.zeros([self.batch_size])
		batch_next_state = np.zeros([self.batch_size, self.state_dim])
		batch_gamma = np.zeros([self.batch_size])
		batch_done = np.zeros([self.batch_size])

		batch_state = self.state_buf[index]
		batch_action = self.action_buf[index]
		batch_reward = self.reward_buf[index]
		batch_next_state = self.next_state_buf[index]
		batch_gamma = self.gamma_buf[index]
		batch_done = self.done_buf[index]

		return {
            'state': batch_state,
			'action': batch_action,
			'reward': batch_reward,
            'next_state': batch_next_state,
			'gamma': batch_gamma,
            'done': batch_done
        }
	
	def get_size(self):
		return self.current_size