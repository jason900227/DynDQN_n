import math
import os
import csv
import random
import torch
import numpy as np
import gym
import argparse
import DQN
import replay_buffer
import state_buffer
import hdbscan
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class Runner:
    def __init__(self, args, env_name, seed):
        self.args = args
        self.env_name = env_name
        self.seed = seed
        self.n_steps = args.n_steps

        self.env = gym.make(env_name, max_episode_steps=1000)
        state, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

        self.env_evaluate = gym.make(env_name, max_episode_steps=1000)  # When evaluating the policy, we need to rebuild an environment
        state, _ = self.env_evaluate.reset(seed=seed)
        self.env_evaluate.action_space.seed(seed)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode

        self.algorithm = 'ESDQN'
        self.replay_buffer = replay_buffer.N_Steps_ReplayBuffer_(args)
        self.state_buffer = state_buffer.State_Buffer(args)
        self.agent = DQN.DQN(args, self.algorithm, self.replay_buffer)

        print("---------------------------------------------------------------------")
        print("Algorithm: {}, Env: {}, Seed: {}, Episode limit: {}".format(self.algorithm, self.env_name, self.seed, self.args.episode_limit))
        print("---------------------------------------------------------------------")

        self.training_episode_data = []
        self.evaluate_epoch_data = []

        self.epsilon = self.args.epsilon_init
        self.epsilon_min = self.args.epsilon_min
        self.epsilon_decay = (self.args.epsilon_init - self.epsilon_min) / self.args.epsilon_decay_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_result(self):
        results_dir = "./results/{}/{}/Training(episode)".format(self.env_name, self.algorithm)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        fields = ['Episode time step', 'Episode number', 'Episode reward', 'Mean Q value', 'Std Q value', 'Cv Q value', 'Mean loss', 'Mean silhouette score']
        filename = os.path.join(results_dir, f"Training(episode)_{self.algorithm}_seed{self.seed}.csv")

        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile, lineterminator='\n')

            # writing the fields
            csvwriter.writerow(fields)

            # writing the data rows
            csvwriter.writerows(self.training_episode_data)

        results_dir = "./results/{}/{}/Evaluate(epoch)".format(self.env_name, self.algorithm)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        fields = ['Epoch time step', 'Epoch number', 'Episode reward']
        filename = os.path.join(results_dir, f"Evaluate(epoch)_{self.algorithm}_seed{self.seed}.csv")

        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile, lineterminator='\n')

            # writing the fields
            csvwriter.writerow(fields)

            # writing the data rows
            csvwriter.writerows(self.evaluate_epoch_data)

    def evaluate_policy(self):
        evaluate_reward = 0
        self.agent.net.eval()

        for i in range(self.args.evaluate_times):
            state, _ = self.env_evaluate.reset(seed=self.seed)
            self.env_evaluate.action_space.seed(self.seed)
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.agent.choose_action(state, epsilon=self.epsilon_min)
                next_state, reward, terminated, truncated, _ = self.env_evaluate.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = torch.tensor(next_state, device=self.device, dtype=torch.float32)

            evaluate_reward += episode_reward

        self.agent.net.train()
        evaluate_reward /= self.args.evaluate_times

        return evaluate_reward

    def pre_fill_replay_buffer(self):
        state, _ = self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        done = False
        t = 0
        episode_steps = 0

        while True:
            t += 1
            episode_steps += 1

            # select action randomly
            action, _ = self.agent.choose_action(state, epsilon=1)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # When dead or win or reaching the max_episode_steps, done will be True, we need to distinguish them;
            # terminal means dead or win, there is no next state s';
            # but when reaching the max_episode_steps, there is a next state s' actually.

            if done and episode_steps != self.args.episode_limit:
                terminal = True
            else:
                terminal = False

            self.replay_buffer.store_transition(state.cpu().numpy(), action, reward, next_state, self.args.gamma, terminal, done)  # Store the transition
            self.state_buffer.store(state.cpu().numpy())  # Store the state
            state = next_state
            state = torch.tensor(state, device=self.device, dtype=torch.float32)

            if done:
                state, _ = self.env.reset(seed=self.seed)
                self.env.action_space.seed(self.seed)
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
                done = False
                episode_steps = 0
                
                if t >= self.args.initial_buffer_capacity:
                    break

        print("Current Replay Buffer Size:{}".format(self.replay_buffer.get_size()))
        print("Current State Buffer Size:{}".format(self.state_buffer.get_size()))    
    
    def run(self):
        # Pre-fill replay buffer with random experiences
        self.pre_fill_replay_buffer()

        state, _ = self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        env_reset = True
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_num = 0
        epoch_num = 0
        step_counter = 0
        discounts_accumulate_reward = 0
        Q_value_list = []
        loss_list = []
        silhouette_list =[]

        progress_bar = tqdm(range(int(self.args.max_training_steps)), desc="Training Progress")

        for t in progress_bar:
            episode_steps += 1
            action, Q_value = self.agent.choose_action(state, epsilon=self.epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # When dead or win or reaching the max_episode_steps, done will be True, we need to distinguish them;
            # terminal means dead or win, there is no next state s';
            # but when reaching the max_episode_steps, there is a next state s' actually.

            if done and episode_steps != self.args.episode_limit:
                terminal = True
            else:
                terminal = False

            if env_reset == True:
                initial_state = state
                initial_action =  action
                env_reset = False
            else:
                state2 = state
                sampled_states = self.state_buffer.sample()
                sampled_states_ = np.vstack([sampled_states, state1.cpu().numpy(), state2.cpu().numpy()])
                sampled_states_hidden_features = self.agent.get_hidden_feature(sampled_states_)
                sampled_states_hidden_features = sampled_states_hidden_features.cpu().numpy()

                # standardized through the removal of the mean and then scaled to unit variance
                scaler = StandardScaler()
                sampled_states_hidden_features_standardized = scaler.fit_transform(sampled_states_hidden_features)

                # Call HDBSCAN to get data labels
                clusterer = hdbscan.HDBSCAN(
                    alpha=1.0,
                    leaf_size=40,
                    min_cluster_size=5,
                    metric="euclidean",
                )
                labels = clusterer.fit_predict(sampled_states_hidden_features_standardized)

                # Calculate HDBSCAN indicator
                mask = labels != -1  # Filter out noise points
                if mask.sum() > 1:  # At least 2 non-noise points are needed to calculate
                    silhouette = silhouette_score(sampled_states_hidden_features_standardized[mask], labels[mask])
                    silhouette_list.append(silhouette)
                    #print("Silhouette Score:", silhouette)

                if (labels[-1] == -1) or (labels[-2] == -1) or (labels[-1] != labels[-2]):  # not similar
                    end_state = state
                    self.replay_buffer.store_transition(initial_state.cpu().numpy(), initial_action, discounts_accumulate_reward, end_state.cpu().numpy(), math.pow(self.args.gamma, (step_counter+1)), end_terminal, end_done)  # Store the transition
                    initial_state = state
                    initial_action = action
                    step_counter = 0
                    discounts_accumulate_reward = 0
                else:  # similar
                    step_counter += 1

            self.state_buffer.store(state.cpu().numpy())  # Store the state
            discounts_accumulate_reward += math.pow(self.args.gamma, step_counter) * reward

            if done:
                end_state = next_state
                end_terminal = terminal
                end_done = done
                self.replay_buffer.store_transition(initial_state.cpu().numpy(), initial_action, discounts_accumulate_reward, end_state, math.pow(self.args.gamma, (step_counter+1)), end_terminal, end_done)  # Store the transition
                step_counter = 0
                discounts_accumulate_reward = 0
                env_reset = True

            state1 = state
            end_terminal = terminal
            end_done = done
            state = next_state
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            Q_value_list.append(Q_value)
            episode_reward += reward

            self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min  # Decay epsilon

            if self.replay_buffer.current_size >= self.args.batch_size:
                loss = self.agent.learn(t+1, self.n_steps)
                loss_list.append(loss)

            if (t+1) % self.args.evaluate_freq == 0:
                epoch_num += 1
                evaluate_reward = self.evaluate_policy()
                tqdm.write("Epoch Num: {} Reward: {}".format(epoch_num, evaluate_reward))
                self.evaluate_epoch_data.append((t+1, epoch_num, evaluate_reward))

            if done:
                episode_num += 1
                mean_silhouette = np.array(silhouette_list).mean()
                tqdm.write("Total Time Step: {} Episode Num: {} Episode Step: {} Reward: {} Mean Silhouette Score: {}".format(t+1, episode_num, episode_steps, episode_reward, mean_silhouette))
                mean_Q_value = np.array(Q_value_list).mean()
                std_Q_value = np.std(np.array(Q_value_list))
                cv_Q_value = std_Q_value / mean_Q_value
                mean_loss = np.array(loss_list).mean()

                self.training_episode_data.append((t+1, episode_num, episode_reward, mean_Q_value, std_Q_value, cv_Q_value, mean_loss, mean_silhouette))

                state, _ = self.env.reset(seed=self.seed)
                self.env.action_space.seed(self.seed)
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
                done = False
                episode_reward = 0
                episode_steps = 0
                silhouette_list = []
                Q_value_list = []
                loss_list = []

        self.save_result()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_training_steps", type=int, default=40000, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=400, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=1, help="Evaluate times")

    parser.add_argument("--replay_buffer_capacity", type=int, default=10000, help="The maximum replay-buffer capacity ")
    parser.add_argument("--state_buffer_capacity", type=int, default=10000, help="The maximum state-buffer capacity ")
    parser.add_argument("--initial_buffer_capacity", type=int, default=500, help='The initial replay-buffer、state-buffer capacity')
    parser.add_argument("--sample_size", type=int, default=500, help="sample size of state-buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate of actor")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=1, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=40000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=False, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=1, help="n_steps")

    args = parser.parse_args()

    # CartPole hyperparameter      Acrobot hyperparameter       Mountain Car hyperparameter
    # hidden_dim = 512             hidden_dim = 24              hidden_dim = 24
    # lr = 0.00025                 lr = 0.0001                  lr = 0.0001
    # target_update_freq = 1000    target_update_freq = 100     target_update_freq = 100
    # max_training_steps = 40000   max_training_steps = 40000   max_training_steps = 300000
    # evaluate_freq = 400          evaluate_freq = 400          evaluate_freq = 3000
    # epsilon_decay_steps = 40000  epsilon_decay_steps = 40000  epsilon_decay_steps = 300000 

    env_names = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0']
    seeds = [0, 10, 20, 40, 60, 80, 100]
    env_index = 0

    for seed in seeds:
        runner = Runner(args=args, env_name=env_names[env_index], seed=seed)
        runner.run()