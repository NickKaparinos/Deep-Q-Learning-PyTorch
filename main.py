"""
Nikos Kaparinos
2023
"""
import time
from os import makedirs
import gymnasium as gym
from dqn_agent import *
from utilities import *


def main():
    """ Main function """
    start = time.perf_counter()
    set_all_seeds()

    # Environment
    # Available environments: 'CartPole-v1' 'MountainCar-v0' 'Acrobot-v1' 'LunarLander-v2'
    env_id = 'LunarLander-v2'
    env = gym.make(env_id, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if 'MountainCar' in env_id:
        env._max_episode_steps = 500

    # Hyperparameters
    agent_hyperparameters = {'lr': 1e-3, 'gamma': 0.999, 'epsilon': 0.8, 'target_update_freq': 250,
                             'buffer_capacity': 100_000, 'batch_size': 128, 'n_neurons': 128}
    learning_hyperparameters = {'max_steps': 500_000, 'fraction_episodes_decay': 0.6, 'max_epsilon': 0.5,
                                'min_epsilon': 0.01, 'step_per_collect': 4, 'update_per_step': 0.5}

    # Agent
    agent = Agent(state_dim=state_dim, action_dim=action_dim, **agent_hyperparameters)

    # Logging
    config = {**agent_hyperparameters, **learning_hyperparameters}
    model_name = f'DQN_{time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime())}'
    log_dir = f'logs/{env_id[:-3]}/{model_name}/'
    makedirs(log_dir, exist_ok=True)
    wandb.init(project=f"DQN-{env_id[:-3]}", entity="nickkaparinos", name=model_name, config=config, notes='',
               reinit=True)

    # Learning
    learn(agent, env, **learning_hyperparameters)

    # Record videos
    record_videos(agent, env, log_dir=log_dir, num_episodes=6)

    # Execution time
    wandb.finish()
    print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")


if __name__ == '__main__':
    main()
