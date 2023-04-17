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


class DQN(nn.Module):
    """ Q Network for DQN Agent """

    def __init__(self, state_dim, action_dim, n_neurons=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, n_neurons)
        self.head = nn.Linear(n_neurons, action_dim)

        self.value_fc1 = nn.Linear(state_dim, n_neurons)
        self.value_head = nn.Linear(n_neurons, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.head(x)

        value = torch.relu(self.value_fc1(state))
        value = self.value_head(value)

        qvals = value + (x - x.mean())
        return qvals


class ReplayBuffer:
    """ Replay Buffer for DQN Agent"""

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
    """ DQN Agent """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.8, target_update_freq=100,
                 buffer_capacity=100_000, batch_size=128, n_neurons=64):
        self.action_dim = action_dim
        self.device = "cpu"
        self.q_network = DQN(state_dim, action_dim, n_neurons=n_neurons).to(self.device)
        self.target_network = DQN(state_dim, action_dim, n_neurons=n_neurons).to(self.device)
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
            return np.random.randint(0, self.action_dim)
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

        q_values_next = self.q_network(next_state)
        _, q_next_argmax = q_values_next.max(1, keepdim=True)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        target_q_values = self.target_network(next_state)
        q_targets = reward + (1 - done) * self.gamma * target_q_values.gather(1, q_next_argmax).squeeze()

        loss = self.loss_fn(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)


def learn(agent, env, max_steps=10000, max_epsilon=1.0, min_epsilon=0.01, fraction_episodes_decay=0.5,
          step_per_collect=50, update_per_step=0.5):
    """ Agent DQN Learning function """
    episode_reward, episode_length = 0, 0
    state = env.reset()[0]
    counter = 0
    for step in tqdm(range(max_steps)):
        # Act
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_length += 1
        counter += 1

        # Save to replay buffer
        agent.push(state, action, reward, next_state, done)
        state = next_state

        # Update agent
        if counter >= step_per_collect:
            for i in range(int(update_per_step * step_per_collect)):
                agent.update()
            counter = 0

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
