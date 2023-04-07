"""
Nikos Kaparinos
2023
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, n_neurons=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.head = nn.Linear(n_neurons, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        # x = torch.relu(self.fc2(x))
        x = self.head(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.8, target_update_freq=100,
                 buffer_capacity=100_000, batch_size=128, n_neurons=64):
        self.device = "cpu"
        self.q_network = DQN(state_dim, action_dim, n_neurons=n_neurons).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(0, 2)
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        q_value = self.q_network(state)
        action = torch.argmax(q_value, dim=1).item()
        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        next_state = torch.tensor(next_state).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        target_q_values = self.target_network(next_state).max(1)[0]
        target = reward + (1 - done) * self.gamma * target_q_values

        loss = self.loss_fn(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)


def learn(agent, env, max_steps=10000, max_epsilon=1.0, min_epsilon=0.01, fraction_episodes_decay=0.5):
    """ Agent DQN Learning function """
    episode_reward, episode_length = 0, 0
    state = env.reset()[0]
    for step in tqdm(range(max_steps)):
        # Act
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Save to replay buffer
        agent.push(state, action, reward, next_state, done)
        state = next_state

        # Update agent
        agent.update()
        episode_reward += reward
        episode_length += 1

        # Logging
        if done:
            wandb.log({'Episode_reward': episode_reward, 'Episode_length': episode_length, 'Step': step})
            episode_reward, episode_length = 0, 0
            state = env.reset()[0]

        # Epsilon scheduling
        if agent.epsilon > min_epsilon:
            agent.epsilon = max_epsilon + step / (max_steps * fraction_episodes_decay) * (min_epsilon - max_epsilon)
        else:
            agent.epsilon = min_epsilon
        wandb.log({'Step': step, 'Epsilon': agent.epsilon})


def set_all_seeds(seed_: int = 0) -> None:
    """ Set all seeds """
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)
    torch.backends.cudnn.deterministic = True  # noqa
