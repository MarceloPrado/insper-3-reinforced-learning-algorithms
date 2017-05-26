'Esse arquivo gera alguns stats dos algoritmos'
import numpy as np
import gym
# from main import random_search
import os


def run_episode(env, params, max_reward, episode, streak):
    'Roda o episodio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
    observation = env.reset()
    totalreward = 0
    for _ in range(max_reward):
        print("\nEpisode = %d" % episode)
        print("t = %d" % _)
        print("Streaks: %d \n" % streak)
        env.render() #para ver treinado
        action = 0 if np.matmul(params, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward



def random_search(env, max_reward, streak_counter):
    '''
    Gera weights aleatorios ate encontrar uma combinacao que
    que satisfaca as condicoes impostas
    '''
    best_params = None
    best_reward = 200
    streak = 0
    episode_counter = 0
    for i_episode in range(1000):
        if streak == 0:
            parameters = np.random.rand(4) * 2 - 1
        else:
            parameters = best_params

        reward = run_episode(env, parameters, max_reward, i_episode, streak)
        
        if reward >= best_reward:
            best_reward = reward
            best_params = parameters
            # caso durou 200 timesteps, considere como resolvido
            if best_reward >= max_reward:
                streak += 1
                episode_counter = i_episode
        
        if reward < max_reward:
            streak = 0

        if streak > streak_counter:
            break
    
    return episode_counter

out = 'gym-out/'
if out:
	if not os.path.exists(out):
		os.makedirs(out)
else:
	if not os.path.exists('gym-out/' + "CartPole-v0"):
		os.makedirs('gym-out/' + "CartPole-v0")
	out = 'gym-out/' + "CartPole-v0"

directory = "gym-out/"
env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env, directory,force=True,video_callable=lambda episode_id: episode_id%100==0)
random_search(env,200,200)
env.close()
gym.scoreboard.api_key = 'sk_bcOLtiCvTKS56VloVRQa6A'
# gym.upload('/Users/marceloprado/cartPoleRL/gym-out')