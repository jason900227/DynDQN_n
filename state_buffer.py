import numpy as np

class State_Buffer:  # store state for ESDQN
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.buffer_capacity = int(args.state_buffer_capacity)
        self.sample_size = args.sample_size
        self.count = 0
        self.current_size = 0
        
        self.state_buf = np.zeros([self.buffer_capacity, self.state_dim])

    def store(self, state):
        self.state_buf[self.count] = state

        self.count = (self.count + 1) % self.buffer_capacity
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.sample_size, replace=False)
        sampled_states = self.state_buf[index]
        
        return sampled_states
    
    def get_size(self):
        return self.current_size