'Esse arquivo gera alguns stats dos algoritmos'
import numpy as np
import gym
# from main import random_search
import os

# def run_episode(env, params, max_reward):
#     'Roda o episodio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
#     print(env)
#     observation = env.reset()
#     totalreward = 0
#     for _ in range(max_reward):
#         # observation = [-0.00903545,  0.04692389, -0.04299039, -0.01087178]
#         # observation = env.reset() -- precisa fazer mas da erro
#         # state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
#         # steps_beyond_done = None
#         # observation = np.array(state)
#         # print (observation)
#         # env.render() #para ver treinado
#         action = 0 if np.matmul(params, observation) < 0 else 1
#         observation, reward, done, info = env.step(action)
#         totalreward += reward
#         if done:
#             break
#     return totalreward

# RUHMAN METHOD - bug
# ele nao ta resetando na linha 31, bem no comeco mesmo. Ele ta usando sempre a 1a observation, nao conseguindo resetar para outras
# ja tentei colocar os metodos de reset antes do for, mas isso causa aquele problema de chamar o reset sem ter encerrado
# talvez precisamos chamar o close
def run_episode(env, params, max_reward,observation):
    'Roda o episodio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
    # observation = env.reset()
    totalreward = 0
    for i in range(max_reward):
        # observation = [-0.00903545,  0.04692389, -0.04299039, -0.01087178]
        # if i !=0:
        #     observation = env.reset()
        # state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        # steps_beyond_done = None
        # observation = np.array(state)
        # print (observation)
        env.render() #para ver treinado
        action = 0 if np.matmul(params, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            print("done")
            break
    
    return totalreward

# def run_episode(env, params, max_reward):
#     'Roda o episodio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
#     state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
#     # env.steps_beyond_done = None
#     # observation = np.array(state)
#     observation = env.reset() 
#     totalreward = 0
#     for _ in range(max_reward):
#         # env.render() #para ver treinado
#         action = 0 if np.matmul(params, observation) < 0 else 1
#         observation, reward, done, info = env.step(action)
#         totalreward += reward
#         if done:
#             break
#     return totalreward

# def run_episode(env, params, max_reward):
#     'Roda o episodio por no max. 200 timesteps, retornanto o totalReward para esse set de params'
#     print("aqui")
#     observation = env.reset()
#     totalreward = 0
#     for _ in range(max_reward):
#         # env.render() #para ver treinado
#         action = 0 if np.matmul(params, observation) < 0 else 1
#         observation, reward, done, info = env.step(action)
#         totalreward += reward
#         print("ali")
#         if done:
#             break
#     return totalreward



def random_search(env, max_reward, streak_counter):
    '''
    Gera weights aleatorios ate encontrar uma combinacao que
    que satisfaca as condicoes impostas
    '''
    best_params = None
    best_reward = 200
    streak = 0
    episode_counter = 0
    for i_episode in range(300):
        print(i_episode)
        print("opa")
        print(streak)
        # observation = env.reset()
        if streak == 0:
            observation = env.reset()
            parameters = np.random.rand(4) * 2 - 1
        else:
            parameters = best_params
        if streak >2:
            observation = env.reset()

        reward = run_episode(env, parameters, max_reward,observation)
        
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
            observation = env.reset()
            break
    
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
env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, directory,force=True,video_callable=lambda episode_id: episode_id%10000==0)
random_search(env,200, 200)
env.close()
gym.upload('/Users/daniruhman/Desktop/Insper 3sem/Robotica/reinforced-learning-algorithms/gym-out',api_key="sk_38r7JkrtRbCqU5vjq6aK6g")