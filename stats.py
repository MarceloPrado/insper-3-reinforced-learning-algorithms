'Esse arquivo gera alguns stats dos algoritmos'
import numpy as np
import gym
from main import random_search, environment

# environment = gym.make("CartPole-v1")

random_search_episodes_to_converge = []
for t in range(1000):
    random_search_episodes_to_converge.append(random_search(environment, 200))

print("Media de episodios necessarios para convergir no random_search: ", np.mean(random_search_episodes_to_converge))