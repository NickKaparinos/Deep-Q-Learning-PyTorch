"""
Nikos Kaparinos
2023
"""
import wandb
from dqn_agent import *
import gymnasium as gym
from os import makedirs
import time


def main():
    """ Main function """
    start = time.perf_counter()
    set_all_seeds()

    # Environment
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters
    agent_hyperparameters = {'lr': 1e-3, 'gamma': 0.99, 'epsilon': 0.8, 'target_update_freq': 100,
                             'buffer_capacity': 100_000, 'batch_size': 128, 'n_neurons': 64}
    learning_hyperparameters = {'max_steps': 500_000, 'fraction_episodes_decay': 0.6, 'max_epsilon': 0.5,
                                'min_epsilon': 0.01, 'step_per_collect': 50, 'update_per_step': 0.5}

    # Agent
    agent = Agent(state_dim=state_dim, action_dim=action_dim, **agent_hyperparameters)

    # Logging
    config = {**agent_hyperparameters, **learning_hyperparameters}
    model_name = f'DQN_{time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime())}'
    log_dir = f'logs/{model_name}/'
    makedirs(log_dir, exist_ok=True)
    wandb.init(project=f"DQN-{env_id[:-3]}", entity="nickkaparinos", name=model_name, config=config, notes='',
               reinit=True)

    # Learning
    learn(agent, env, **learning_hyperparameters)

    # Execution time
    wandb.finish()
    print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")
    fet = 5


if __name__ == '__main__':
    main()
