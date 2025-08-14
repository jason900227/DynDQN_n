import torch
import torch.nn as nn
import numpy as np
import copy

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, s, return_hidden=False):
        s = torch.relu(self.fc1(s))
        hidden_features = torch.relu(self.fc2(s))
        Q = self.fc3(hidden_features)
        
        if return_hidden:
            return hidden_features
        else:
            return Q

class DQN(object):
    def __init__(self, args, algorithm, replay_buffer):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.max_train_steps = args.max_training_steps
        self.lr = args.lr
        self.use_lr_decay = args.use_lr_decay
        self.grad_clip = args.grad_clip
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0

        self.algorithm = algorithm
        self.replay_buffer = replay_buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(args).to(self.device)
        self.target_net = copy.deepcopy(self.net).to(self.device)  # Copy the online_net to the target_net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def get_hidden_feature(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            hidden_features = self.net(state, return_hidden=True)

        return hidden_features

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            state = torch.unsqueeze(state.clone().detach(), 0)
            q = self.net(state)

            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)

            q = q[0, action].item()

            return action, q

    def learn(self, total_steps, n_steps):

        if self.algorithm == "MMDQN":
            avg_q_target = None  
            indices = self.replay_buffer.sample_indices(n_steps)

            for n in range(1, n_steps+1):
                sample = self.replay_buffer.sample(n, indices)
                state = torch.tensor(sample['state'], dtype=torch.float32).to(self.device)
                action = torch.tensor(sample['action'], dtype=torch.long).to(self.device)
                reward = torch.tensor(sample['reward'], dtype=torch.float32).to(self.device).view(-1, 1)
                next_state = torch.tensor(sample['next_state'], dtype=torch.float32).to(self.device)
                done = torch.tensor(sample['done'], dtype=torch.float32).to(self.device).view(-1, 1)
                gamma = torch.tensor(sample['gamma'], dtype=torch.float32).to(self.device).view(-1, 1)

                with torch.no_grad():  # q_target has no gradient
                    q_target = reward + gamma * (1 - done) * self.target_net(next_state).max(dim=-1)[0]  # shape：(batch_size,)

                if avg_q_target is None:
                    avg_q_target = q_target.detach()
                else:
                    avg_q_target += q_target.detach()

            avg_q_target /= n_steps
            q_current = self.net(state).gather(1, action).squeeze(1)  # shape：(batch_size,)
            td_errors = q_current - avg_q_target  # shape：(batch_size,)
            loss = (td_errors ** 2).mean()

        else:
            indices = self.replay_buffer.sample_indices(n_steps)
            sample = self.replay_buffer.sample(n_steps, indices)
            state = torch.tensor(sample['state'], dtype=torch.float32).to(self.device)
            action = torch.tensor(sample['action'], dtype=torch.long).to(self.device)
            reward = torch.tensor(sample['reward'], dtype=torch.float32).to(self.device).view(-1, 1)
            next_state = torch.tensor(sample['next_state'], dtype=torch.float32).to(self.device)
            done = torch.tensor(sample['done'], dtype=torch.float32).to(self.device).view(-1, 1)
            gamma = torch.tensor(sample['gamma'], dtype=torch.float32).to(self.device).view(-1, 1)

            with torch.no_grad():
                q_target = reward + gamma * (1 - done) * self.target_net(next_state).max(dim=-1)[0]  # shape：(batch_size,)

            q_current = self.net(state).gather(1, action).squeeze(1)  # shape：(batch_size,)
            td_errors = q_current - q_target  # shape：(batch_size,)
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

        return loss.item()

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now