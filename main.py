"Esse arquivo testa diferentes modelos de RL para o enviroment de cartPole"
import gym
import numpy as np

def run_episode(env, params, max_reward):
    'Roda o episódio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
    observation = env.reset()
    totalreward = 0
    for _ in range(max_reward):
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
    best_reward = 0
    episode_counter = 0
    for i_episode in range(10000):
        episode_counter = i_episode
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(environment, parameters, max_reward)
        if reward > best_reward:
            best_reward = reward
            best_params = parameters
            # caso durou 200 timesteps, considere como resolvido
            if best_reward == max_reward:
                break
    # print("Best params:", best_params, "\nReward: ", best_reward)
    return episode_counter


environment = gym.make("CartPole-v1")

