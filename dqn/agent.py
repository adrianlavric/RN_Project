import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os
import torch.nn.functional as F
from collections import deque

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
device = 'cpu'

LOG_FILE   = os.path.join(RUNS_DIR, 'v2.log')
MODEL_FILE = os.path.join(RUNS_DIR, 'v2.pt')
GRAPH_FILE = os.path.join(RUNS_DIR, 'v2.png')

class ReplayBuffer():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

            nn.init.kaiming_uniform_(self.fc_value.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc_advantages.weight, nonlinearity='relu')
        else:
            self.output = nn.Linear(hidden_dim, action_dim)
            nn.init.kaiming_uniform_(self.output.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q

class Agent():

    def __init__(self):
        self.learning_rate_a    = 0.00025 
        self.discount_factor_g  = 0.99 
        self.network_sync_rate  = 500  
        self.replay_memory_size = 100000  
        self.mini_batch_size    = 32    
        self.epsilon_init       = 1      
        self.epsilon_decay      = 0.9995       
        self.epsilon_min        = 0.01      
        self.stop_on_reward     = 5000     
        self.fc1_nodes          = 256 
        self.enable_double_dqn  = True
        self.loss_fn = nn.MSELoss()          
        self.optimizer = None           

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Start:"
            print(log_message)
            with open(LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make('FlappyBird-v0', render_mode='human' if render else None, use_lidar=False)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayBuffer(self.replay_memory_size)

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []

            step_count=0
            best_reward = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False    
            episode_reward = 0.0 

            while(not terminated and episode_reward < self.stop_on_reward):

                if is_training and random.random() < epsilon:
                  action = env.action_space.sample()
                  action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                  with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state,reward,terminated,truncated,info = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count+=1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
              if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, model saved."
                print(log_message)
                with open(LOG_FILE, 'a') as file:
                  file.write(log_message + '\n')

                torch.save(policy_dqn.state_dict(), MODEL_FILE)
                best_reward = episode_reward

              current_time = datetime.now()
              if current_time - last_graph_update_time > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time

              if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                  target_dqn.load_state_dict(policy_dqn.state_dict())
                  step_count=0


    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) 
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122) 
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
      states, actions, new_states, rewards, terminations = zip(*mini_batch)
      states = torch.stack(states)
      actions = torch.stack(actions)
      new_states = torch.stack(new_states)
      rewards = torch.stack(rewards)
      terminations = torch.tensor(terminations).float().to(device)

      with torch.no_grad():
        if self.enable_double_dqn:
          best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

          target_q = rewards + (1-terminations) * self.discount_factor_g * \
                  target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
        else:
          target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

      current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
      loss = self.loss_fn(current_q, target_q)
      self.optimizer.zero_grad()
      loss.backward()            
      self.optimizer.step()      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent()
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)