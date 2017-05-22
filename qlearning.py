import random
import copy
from collections import defaultdict
from collections import deque
from time import sleep

from collections import namedtuple
import numpy as np
import math

class QLearning():
    def __init__(self, env):
        self.env = env
        # Number of discrete states (bucket) per state dimension
        self.NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
        # Number of discrete actions
        self.NUM_ACTIONS = env.action_space.n # (left, right)
        # Bounds for each discrete state
        self.STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

        self.STATE_BOUNDS[1] = [-0.5, 0.5]
        self.STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
        # Index of the action
        self.ACTION_INDEX = len(self.NUM_BUCKETS)

        ## Creating a Q-Table for each state-action pair
        self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,))

        
        ## Learning related constants
        self.MIN_EXPLORE_RATE = 0.01
        self.MIN_LEARNING_RATE = 0.1

        ## Defining the simulation related constants
        self.NUM_EPISODES = 1000
        self.MAX_T = 250
        self.STREAK_TO_END = 120
        self.SOLVED_T = 199
        self.DEBUG_MODE = True

    
    def get_explore_rate(self,t):
        return max(self.MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

    def get_learning_rate(self,t):
        return max(self.MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

    def simulate(self):
        ## Instantiating the learning related parameters
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        discount_factor = 0.99  # since the world is unchanging

        num_streaks = 0

        for episode in range(self.NUM_EPISODES):

            # Reset the environment
            observation = self.env.reset()

            # Get the initial state matrix
            state_0 = self.state_to_bucket(observation)

            for t in range(self.MAX_T):
                # self.env.render()

                # Select an action
                action = self.select_action(state_0, explore_rate)

                # Execute the action
                observation, reward, done, _ = self.env.step(action)

                # Observe the result
                state = self.state_to_bucket(observation)

                # Update the Q based on the result
                best_q = np.amax(self.q_table[state])
                self.q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - self.q_table[state_0 + (action,)])

                # Setting up for the next iteration
                state_0 = state

                # Print data
                if (self.DEBUG_MODE):
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("Reward: %f" % reward)
                    print("Best Q: %f" % best_q)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)

                    print("")

                if done:                
                    print("Episode %d finished after %f time steps" % (episode, t))
                if (t >= self.SOLVED_T):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

                sleep(0.25)

            # It's considered done when it's solved over 120 times consecutively
            if num_streaks > self.STREAK_TO_END:
                break

            # Update parameters
            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)


    def select_action(self,state, explore_rate):
        # Select a random action
        if random.random() < explore_rate:
            action = self.env.action_space.sample()
        # Select the action with the highest q
        else:
            action = np.argmax(self.q_table[state])
        return action



    def state_to_bucket(self,state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= self.STATE_BOUNDS[i][1]:
                bucket_index = self.NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.STATE_BOUNDS[i][1] - self.STATE_BOUNDS[i][0]
                offset = (self.NUM_BUCKETS[i]-1)*self.STATE_BOUNDS[i][0]/bound_width
                scaling = (self.NUM_BUCKETS[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)