import gym
import numpy as np
import random
import math
from time import sleep
import os


out = 'gym-out'

if out:
	if not os.path.exists(out):
		os.makedirs(out)
else:
	if not os.path.exists('gym-out/' + "CartPole-v0"):
		os.makedirs('gym-out/' + "CartPole-v0")
	out = 'gym-out/' + "CartPole-v0"

directory = "gym-out/"

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, directory,force=True)


# Simulation variables
amount_of_episodes = 1000
episode_timestep = 250
streak_limit = 120
solved_timestep = 199
is_debugging = True

# Number of discrete 'gaps' per state
states_size = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of possible actions
actions_size = env.action_space.n # (left, right)

# Limits for each discrete state
state_limit = list(zip(env.observation_space.low, env.observation_space.high))
state_limit[1] = [-0.5, 0.5]
state_limit[3] = [-math.radians(50), math.radians(50)]

# Q learning variables
q_table = np.zeros(states_size + (actions_size,))
exploration_rate = 0.03
learning_rate = 0.08


def simulate():
    learning_rate = get_learning_rate(0)
    explore_rate = get_exploration_rate(0)
    gamma = 0.99

    num_streaks = 0

    for episode in range(amount_of_episodes):

        # Reset the environment
        obv = env.reset()

        # initial state
        state_0 = state_to_discrete_value(obv)
        for t in range(episode_timestep):
            env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_discrete_value(obv)

            # Update q table based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + gamma*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            if (is_debugging):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d \n" % num_streaks)

            if done:
               print("Episode %d finished after %f time steps" % (episode, t))
               if (t >= solved_timestep):
                   num_streaks += 1
               else:
                   num_streaks = 0
               break

        if num_streaks > streak_limit:
            break

        # Update parameters
        explore_rate = get_exploration_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q-value
    else:
        action = np.argmax(q_table[state])
    return action


def get_exploration_rate(t):
    return max(exploration_rate, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(learning_rate, min(0.5, 1.0 - math.log10((t+1)/25)))

def state_to_discrete_value(state):
    discrete_list = []
    for i in range(len(state)):
        if state[i] <= state_limit[i][0]:
            discrete_index = 0
        elif state[i] >= state_limit[i][1]:
            discrete_index = states_size[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = state_limit[i][1] - state_limit[i][0]
            offset = (states_size[i]-1)*state_limit[i][0]/bound_width
            scaling = (states_size[i]-1)/bound_width
            discrete_index = int(round(scaling*state[i] - offset))
        discrete_list.append(discrete_index)
    return tuple(discrete_list)

if __name__ == "__main__":
    simulate()
    env.close()
    gym.scoreboard.api_key = 'sk_bcOLtiCvTKS56VloVRQa6A'
    gym.upload('/Users/marceloprado/cartPoleRL/gym-out')

