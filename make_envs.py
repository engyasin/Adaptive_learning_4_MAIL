#from pettingzoo.mpe import simple_reference_v2
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_spread_v2
import supersuit as ss
from stable_baselines3 import A2C, PPO, SAC, TD3
from multi_ppo import ppo_2_multi , MA_ActCrit, MA_ActorCritic

from stable_baselines3.common.monitor import Monitor
from envs_utils import unified_env,VecExtractDictObs,Vecbooldone
import numpy as np


def make_env_s(n_agents,max_cycles,num_ma_env=8,seed=42):
    env = simple_spread_v2.parallel_env(N=n_agents, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False)
    env.reset(seed=seed)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)

    return Vecbooldone(n_agents,ss.concat_vec_envs_v1(env_, num_ma_env, num_cpus=4, base_class='stable_baselines3'))

def make_env_adv_s(n_agents,max_cycles,num_ma_env=8,seed=42):
    env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycles, continuous_actions=False)
    env.reset(seed=seed)
    env=ss.pad_observations_v0(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)
    return Vecbooldone(n_agents+1,ss.concat_vec_envs_v1(env_, num_ma_env, num_cpus=4, base_class='stable_baselines3'))


def make_env_s_uni(n_agents,max_cycles):
    env = simple_spread_v2.parallel_env(N=n_agents, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)
    #excatly 1 env to work
    env_ = ss.concat_vec_envs_v1(env_, 1 ,num_cpus=16, base_class="stable_baselines3")
    return VecExtractDictObs(env_)

def make_env_adv_s_uni(n_agents,max_cycles):
    env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycles, continuous_actions=False)
    #env.reset(seed=SEED)
    env = ss.pad_observations_v0(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)
    env_ = ss.concat_vec_envs_v1(env_, 1,num_cpus=4, base_class="stable_baselines3")

    return VecExtractDictObs(env_)


def make_env_adv_s_uni_g(n_agents,max_cycles):
    env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycles, continuous_actions=False)
    g = env.unwrapped.world.agents[1].goal_a.state.p_pos
    #env.reset(seed=SEED)
    env = ss.pad_observations_v0(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)
    env_ = ss.concat_vec_envs_v1(env_, 1,num_cpus=4, base_class="stable_baselines3")

    return VecExtractDictObs(env_),g

def make_adv_env_s_uni(n_agents=3,max_cycle=50):
    env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycle, continuous_actions=False)
    env.reset(seed=42)
    env2 = ss.pad_observations_v0(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env2)
    env_ = ss.concat_vec_envs_v1(env_, 1,num_cpus=4, base_class="stable_baselines3")

    return VecExtractDictObs(env_)

def make_adv_env_s(n_agents=3,max_cycle=50):
    env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycle, continuous_actions=False)
    #env = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=max_cycle, continuous_actions=False)
    env.reset(seed=42)
    env2 = ss.pad_observations_v0(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env2)

    return ss.concat_vec_envs_v1(env_, 1, num_cpus=4, base_class='stable_baselines3')


def grouped_env(SEED,n_agents,max_cycles,local_r=0.5):
    env = simple_spread_v2.parallel_env(N=n_agents, local_ratio=local_r, max_cycles=max_cycles, continuous_actions=False)
    env.reset(seed=SEED)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)
    return Monitor(unified_env(env_))

def grouped_env_adv(SEED,n_agents,max_cycles,local_r=0.5):
    env = simple_adversary_v2.parallel_env(N=n_agents, max_cycles=max_cycles, continuous_actions=False)
    env.reset(seed=SEED)
    env=ss.pad_observations_v0(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)
    return Monitor(unified_env(env_))




def train_expert_ma(env, n_agents):
    policy_kwargs = dict( net_arch=[64,dict(pi=[32,32], vf=[32,32])])
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
        tensorboard_log='logs/',
        #policy_kwargs=policy_kwargs
    )
    expert.learn(total_timesteps=10000) # TODO
    return expert

from stable_baselines3.ppo import MlpPolicy

def train_expert(env):
    policy_kwargs = dict( net_arch=[64,dict(pi=[32,32], vf=[32,32])])
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
        tensorboard_log='logs/',
        #policy_kwargs=policy_kwargs,
    )
    expert.learn(total_timesteps=10000) # TODO
    return expert
