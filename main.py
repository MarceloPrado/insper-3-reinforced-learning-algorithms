"Esse arquivo testa diferentes modelos de RL para o enviroment de cartPole"
import gym
import numpy as np
from qlearning import QLearning
import os

def run_episode(env, params, max_reward):
    'Roda o episodio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
    observation = env.reset()
    totalreward = 0
    for _ in range(max_reward):
        # observation = [-0.00903545,  0.04692389, -0.04299039, -0.01087178]
        # observation = env.reset() -- precisa fazer mas da erro
        state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        steps_beyond_done = None
        observation = np.array(state)
        # print (observation)
        # env.render() #para ver treinado
        action = 0 if np.matmul(params, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward



def random_search(env, max_reward):
    '''
    Gera weights aleatorios ate encontrar uma combinacao que
    que satisfaca as condicoes impostas
    '''
    best_params = None
    best_reward = 200
    episode_counter = 0
    for i_episode in range(30000):
        episode_counter = i_episode
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters, max_reward)
        if reward > best_reward:
            best_reward = reward
            best_params = parameters
            # caso durou 200 timesteps, considere como resolvido
            if best_reward == max_reward:
                break
    # print("Best params:", best_params, "\nReward: ", best_reward)
    return episode_counter
out = 'gym/out'

if out:
	if not os.path.exists(out):
		os.makedirs(out)
else:
	if not os.path.exists('gym-out/' + "CartPole-v1"):
		os.makedirs('gym-out/' + "CartPole-v1")
	out = 'gym-out/' + "CartPole-v1"

directory = "gym-out/"
# environment = gym.make("CartPole-v1")
# print("opa3")
# environment = gym.wrappers.Monitor(environment, directory,force=True)
# environment.close()
# gym.upload('gym-out/')