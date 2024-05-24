#from pettingzoo.mpe import simple_reference_v2
from pettingzoo.mpe import simple_spread_v2,simple_adversary_v2
import time
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
import supersuit as ss
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy

import torch as th


from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import  networks

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
from multi_ppo import ppo_2_multi , MA_ActCrit, MA_ActorCritic

from expert import expert_p, expert_2_multi
from envs_utils import unified_obs, VecExtractDictObs,Vecbooldone,Vecbooldone_Soccer

import time
import gym
from gym import register
from petting_zoo_v import parallel_soccer



register(
    id='multigrid-soccer-v0',
    entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
    #kwargs={'seed':int(time.time()) },
)


def make_env_soccer(n_agents,max_cycles,num_ma_env=1,seed=42):
    env = parallel_soccer(lambda: gym.make('multigrid-soccer-v0'),n_agents=n_agents,  max_cycles=max_cycles)
    #env.reset(seed=seed)
    num_cpus = 1
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)

    return Vecbooldone_Soccer(n_agents,ss.concat_vec_envs_v1(env_, num_ma_env, num_cpus=num_cpus, base_class='stable_baselines3'))

def make_make_env(env_type,ma=False):

    make_env = None

    if env_type == 'spread':

        if ma:
            def make_env(n_agents=4,max_cycles=50):
                env = simple_spread_v2.parallel_env(N=n_agents, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False)
                env_ = ss.pettingzoo_env_to_vec_env_v1(env)
                env_ = ss.concat_vec_envs_v1(env_, 1,num_cpus=4, base_class="stable_baselines3")
                return VecExtractDictObs(env_)
        else:
            def make_env(n_agents=4,max_cycles=50):
                env = simple_spread_v2.parallel_env(N=n_agents, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False)
                env_ = ss.pettingzoo_env_to_vec_env_v1(env)

                return Vecbooldone(n_agents,ss.concat_vec_envs_v1(env_, 1, num_cpus=4, base_class='stable_baselines3'))

    elif env_type == 'reference':
        print('Reference Env not implemented becuase it has only 2 agents')

    elif env_type == 'adversary':

        if ma:
            def make_env(n_agents=3,max_cycles=50):
                env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycles, continuous_actions=False)
                env.reset(seed=42)
                env2 = ss.pad_observations_v0(env)
                env_ = ss.pettingzoo_env_to_vec_env_v1(env2)
                env_ = ss.concat_vec_envs_v1(env_, 1,num_cpus=4, base_class="stable_baselines3")

                return VecExtractDictObs(env_)
        else:
            def make_env(n_agents=3,max_cycles=50):
                env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycles, continuous_actions=False)
                #env = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=max_cycle, continuous_actions=False)
                env.reset(seed=42)
                env2 = ss.pad_observations_v0(env)
                env_ = ss.pettingzoo_env_to_vec_env_v1(env2)
                env3 = ss.concat_vec_envs_v1(env_, 1, num_cpus=4, base_class='stable_baselines3')
                #breakpoint()

                return Vecbooldone(n_agents+1,env3)
    
    return make_env

def train_expert_ma(env, n_agents):
    # Create the agent
    def policy_(observation_space,action_space,lr_schedule,n_agents=n_agents, **kwargs):
        return MA_ActorCritic(observation_space,action_space,lr_schedule,n_agents=n_agents, **kwargs)

    expert = PPO(
        #n_agents= n_agents,
        policy=policy_,
        env=env,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
        tensorboard_log='logs/'
        #policy_kwargs=policy_kwargs,
    )
    expert.learn(total_timesteps=8000) # TODO
    return expert

def train_expert(env, n_agents):
    policy_kwargs = dict( net_arch=[32,dict(pi=[32,32], vf=[32,32])])
    # Create the agent
    expert = PPO(
        #n_agents= n_agents,
        policy=MlpPolicy,
        env=env,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
        tensorboard_log='logs/'
        #policy_kwargs=policy_kwargs,
    )
    expert.learn(total_timesteps=8000) # TODO
    return expert

def compare_actions(a1,a2):
    # simple matching crtiera
    return a1==a2

def compare_x(policy_1,policy_2,n_agents=8,env_type='spread',ma=True):

    # 1. run the same environment.
    # 2. substract the actions (not state) for every step. 
    # (try do action then undo if there's such a thing)
    # 3. step with the original expert
    # 4. do the same for the new state
    # 5. after the known number of steps (fixed) calculte the mean difference.


    #ma = True
    max_cycles = 15
    num_of_episodes = 15

    make_env = make_make_env(env_type=env_type,ma=ma)# or pass the env
    env = make_env(n_agents=n_agents,max_cycles=max_cycles)

    if False:
        if ma:
            model = train_expert_ma(make_env(n_agents=n_agents),n_agents)
        else:
            model = train_expert(make_env(n_agents=n_agents),n_agents)
    obs = env.reset()

    result = []

    for i in range(max_cycles*num_of_episodes):

        actions_2,_ = policy_2.predict(obs,deterministic=True)
        if ma:
            obs = obs.reshape(obs.shape[0],-1)

        actions_1,_ = policy_1.predict(obs,deterministic=True)
        if ma: actions_1 = actions_1[0]
        #[env.action_space.sample() for _ in range(env.num_envs)]
        #ob,re,do,inf = env.step()
        #breakpoint()
        #TODO compare how traj by states not actions (positions)
        result.append(compare_actions(actions_1,actions_2))

        obs ,rewards ,dones ,infos = env.step(actions_1)
        if dones.all():
            obs = env.reset()
        #env.render()
        #time.sleep(0.1)
    
    return np.mean(result)

def compare_xx2(policy_1,policy_2,n_agents=8,env_type='spread',ma=True):
    # policy_1 is the expert

    # 1. run the same environment.
    # 2. substract the actions (not state) for every step. 
    # (try do action then undo if there's such a thing)
    # 3. step with the original expert
    # 4. do the same for the new state
    # 5. after the known number of steps (fixed) calculte the mean difference.


    #ma = True
    max_cycles = 15*2 # to allow for double pass
    num_of_episodes = 100
    
    #NOTE hard coded:
    old_shape = None
    if env_type == 'soccer' :
        max_cycles = 15*2
        num_of_episodes = 100
        env = make_env_soccer(n_agents=n_agents,max_cycles=max_cycles)
        old_shape = (-1,5,5,6)
    else:
        make_env = make_make_env(env_type=env_type,ma=ma)# or pass the env
        env = make_env(n_agents=n_agents,max_cycles=max_cycles)

    if False:
        if ma:
            model = train_expert_ma(make_env(n_agents=n_agents),n_agents)
        else:
            model = train_expert(make_env(n_agents=n_agents),n_agents)
    #obs = env.reset()
    

    ADEs = []
    FDEs = []
    eps_rewards = []
    for i in range(num_of_episodes):

        c_seed = int(time.time())
        obs_ = env.reset(seed=c_seed)
        #f_state = env.get_state()[:]
        #f_state = [(row[2:4],np.array([0.,0.])) for row in obs_]
        if env_type != 'soccer':
            f_state = env.get_state().copy()
        learner_traj = []

        ep_rewards = []
        obs = obs_.copy()
        for j in range((max_cycles-2)//2):
            if old_shape:
                obs = obs.reshape(*old_shape)
            actions_2,_ = policy_2.predict(obs,deterministic=True)
            #breakpoint()
            obs ,rewards ,dones ,infos = env.step(actions_2)
            if env_type != 'soccer':
                learner_traj.append(obs[:,2:4]) #positions
            else:
                learner_traj.append(env.get_positions())
            ep_rewards.append(rewards/n_agents)

        obs = obs_.copy()
        if env_type != 'soccer':
            env.reset_set_state(f_state.copy())
        else:
            obs__ = env.reset(seed=c_seed)
            #print('is 0: ',(obs_-obs__).sum())
        #obs ,rewards ,dones ,infos = env.step(np.zeros_like(actions_2))#empty action
        expert_traj = []
        for j in range((max_cycles-2)//2):
            if old_shape:
                obs = obs.reshape(*old_shape)
            actions_1,_ = policy_1.predict(obs,deterministic=True)
            obs ,rewards ,dones ,infos = env.step(actions_1)
            if env_type != 'soccer':
                expert_traj.append(obs[:,2:4]) #positions
            else:
                expert_traj.append(env.get_positions())

        traj_diff = np.vstack(expert_traj) - np.vstack(learner_traj)
        ADEs.append(np.linalg.norm(traj_diff,axis=1).mean())
        FDEs.append(np.linalg.norm(traj_diff[-n_agents:],axis=1).mean())
        eps_rewards.append(np.array(ep_rewards).flatten().sum())
        #env.render()
        #time.sleep(0.1)
    return np.mean(ADEs),np.mean(FDEs),np.mean(eps_rewards),np.std(eps_rewards)
def compare_xx(policy_1,policy_2,n_agents=8,env_type='spread',ma=True):
    # policy_1 is the expert

    # 1. run the same environment.
    # 2. substract the actions (not state) for every step. 
    # (try do action then undo if there's such a thing)
    # 3. step with the original expert
    # 4. do the same for the new state
    # 5. after the known number of steps (fixed) calculte the mean difference.


    #ma = True
    max_cycles = 15*2 # to allow for double pass
    num_of_episodes = 100
    
    #NOTE hard coded:
    old_shape = None
    if env_type == 'soccer' :
        max_cycles = 15*2
        num_of_episodes = 500
        env = make_env_soccer(n_agents=n_agents,max_cycles=max_cycles)
        old_shape = (-1,5,5,6)
    else:
        make_env = make_make_env(env_type=env_type,ma=ma)# or pass the env
        env = make_env(n_agents=n_agents,max_cycles=max_cycles)

    if False:
        if ma:
            model = train_expert_ma(make_env(n_agents=n_agents),n_agents)
        else:
            model = train_expert(make_env(n_agents=n_agents),n_agents)
    #obs = env.reset()
    

    ADEs = []
    FDEs = []
    eps_rewards = []
    for i in range(num_of_episodes):

        c_seed = int(time.time())
        obs_ = env.reset(seed=c_seed)
        #f_state = env.get_state()[:]
        #f_state = [(row[2:4],np.array([0.,0.])) for row in obs_]
        if env_type != 'soccer':
            f_state = env.get_state().copy()
        learner_traj = []

        ep_rewards = []
        obs = obs_.copy()
        for j in range((max_cycles-2)//2):

            actions_2,_ = policy_2.predict(obs,deterministic=True)
            #breakpoint()
            obs ,rewards ,dones ,infos = env.step(actions_2)
            if env_type != 'soccer':
                learner_traj.append(obs[:,2:4]) #positions
            else:
                learner_traj.append(env.get_positions())
            ep_rewards.append(rewards/n_agents)

        obs = obs_.copy()
        if env_type != 'soccer':
            env.reset_set_state(f_state.copy())
        else:
            obs__ = env.reset(seed=c_seed)
            #print('is 0: ',(obs_-obs__).sum())
        #obs ,rewards ,dones ,infos = env.step(np.zeros_like(actions_2))#empty action
        expert_traj = []
        for j in range((max_cycles-2)//2):
            if old_shape:
                obs = obs.reshape(*old_shape)
            actions_1,_ = policy_1.predict(obs,deterministic=True)
            obs ,rewards ,dones ,infos = env.step(actions_1)
            if env_type != 'soccer':
                expert_traj.append(obs[:,2:4]) #positions
            else:
                expert_traj.append(env.get_positions())

        traj_diff = np.vstack(expert_traj) - np.vstack(learner_traj)
        ADEs.append(np.linalg.norm(traj_diff,axis=1).mean())
        FDEs.append(np.linalg.norm(traj_diff[-n_agents:],axis=1).mean())
        eps_rewards.append(np.array(ep_rewards).flatten().sum())
        #env.render()
        #time.sleep(0.1)

    del env
    r,std = np.mean(eps_rewards),np.std(eps_rewards)
    if env_type == 'soccer' :
        r,std = np.std(ADEs),np.std(FDEs)
    return np.mean(ADEs),np.mean(FDEs),r,std


def compare_m_states(policy_1,policys_list,env_type='spread',n_agents=8,max_cycles=50,ma=False):

    #ma = True
    n_polices = len(policys_list)+1
    max_cycles_ = max_cycles*n_polices*2
    # TODO make env
    make_env = make_make_env(env_type=env_type,ma=ma)# or pass the env
    env = make_env(n_agents=n_agents,max_cycles=max_cycles_)

    num_of_episodes = 70

    ADEs = [[] for _ in policys_list]
    FDEs = [[] for _ in policys_list]
    Rs   = [[] for _ in range(n_polices)]

    for i in range(num_of_episodes):

        obs_ = env.reset()
        f_state = env.get_state().copy()
        #f_state = [(row[2:4],np.array([0.,0.])) for row in obs_]

        learner_traj = [[] for _ in policys_list]
        obs = obs_.copy()

        for m,policy_2 in enumerate(policys_list):

            for j in range((max_cycles_-n_polices)//n_polices):

                actions_2,_ = policy_2.predict(obs,deterministic=True)
                obs ,rewards ,dones ,infos = env.step(actions_2)

                Rs[m].append(rewards.mean())
                learner_traj[m].append(obs[:,2:4]) #positions

            obs = obs_.copy()
            env.reset_set_state(f_state.copy())

        #obs ,rewards ,dones ,infos = env.step(np.zeros_like(actions_2))#empty action
        expert_traj = []
        for j in range((max_cycles_-n_polices)//n_polices):

            actions_1,_ = policy_1.predict(obs,deterministic=True)
            obs ,rewards ,dones ,infos = env.step(actions_1)
            Rs[-1].append(rewards.mean())
            expert_traj.append(obs[:,2:4]) #positions

        for m in range(len(policys_list)):
            traj_diff = np.vstack(expert_traj) - np.vstack(learner_traj[m])
            ADEs[m].append(np.linalg.norm(traj_diff,axis=1).mean())
            FDEs[m].append(np.linalg.norm(traj_diff[-n_agents:],axis=1).mean())

    #reward is wrong
    rewards_ = [(np.round(np.mean(r)*max_cycles,2),np.round(np.std(r)*max_cycles,2)) for r in Rs]
    return [np.mean(ade) for ade in ADEs],[np.mean(fde) for fde in FDEs], rewards_


def compare_m_actions(policy_1,policys_list,env,n_agents=8,max_cycles=50):

    # 1. run the same environment.
    # 2. substract the actions (not state) for every step. 
    # (try do action then undo if there's such a thing)
    # 3. step with the original expert
    # 4. do the same for the new state
    # 5. after the known number of steps (fixed) calculte the mean difference.

    ma = True
    num_of_episodes = 15

    if False:
        if ma:
            model = train_expert_ma(make_env(n_agents=n_agents),n_agents)
        else:
            model = train_expert(make_env(n_agents=n_agents),n_agents)
    
    obs = env.reset()

    result = [[] for _ in policys_list]

    for i in range(max_cycles*num_of_episodes):


        if ma:
            obs_ = obs.reshape(obs.shape[0],-1)
            actions_1_s,_ = policy_1.predict(obs_,deterministic=True)
            actions_1 = actions_1_s[0]
        else:
            actions_1_s,_ = policy_1.predict(obs,deterministic=True)
            actions_1 = actions_1_s

        #actions_2 = []
        for i,policy_2 in enumerate(policys_list):
            action_2,_ = policy_2.predict(obs,deterministic=True)
            #actions_2.append(action_2)

            #[env.action_space.sample() for _ in range(env.num_envs)]
            #ob,re,do,inf = env.step()
            #breakpoint()
            result[i].append(compare_actions(actions_1,action_2))

        obs ,rewards ,dones ,infos = env.step(actions_1)
        if dones.all():
            obs = env.reset()
        #env.render()
        #time.sleep(0.1)
    
    return [np.mean(r) for r in result]