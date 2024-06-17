
import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gc

from pettingzoo.mpe import simple_spread_v2,simple_adversary_v2
from pettingzoo.butterfly import pistonball_v6

import supersuit as ss
from stable_baselines3.common.monitor import Monitor
from multi_ppo import ppo_2_multi, unified_Mlp , MA_ActorCritic, MA_Disc,ppo_compete_ma

#from make_envs import unified_env, Vec_bool_done, VecRolloutReward
from make_envs import grouped_env_adv,make_env_adv_s,make_env_s_uni

import numpy as np
from compare_policies import compare_x,compare_xx

from imitation.data import rollout,types
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.util.util import make_vec_env
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL

from gail_new import GAIL_ENS,AIRL_ENS
from imitation.rewards.reward_nets import BasicRewardNet,BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import  networks
from stable_baselines3.common.evaluation import evaluate_policy

from expert import expert_p,expert_2_multi,expert_adv,expert_2_multi_adv
from ensemble import from_many_to_single, convert_batch, DsEnsemble4,Shaped_R_Blended
import tqdm

import random


def main(num_of_exp_eps=32,test_mode=0):

    #TODO 
    #1 use cuda
    #2 use better hyperparameters (example batch)
    #3 compare_x
    #4 envs

    SEED = 42
    env_name = 'adversary'
    local_r = 0.5
    debug = False
    warm_start = True
    #test_mode = 0#[0: full, 1: single, 2: group]

    max_cycles = 15

    #num_of_exp_eps = 256 # [25,100,400]
    n_agents = 3 # [3,4]


    lr = 0.001 # [0.003,0.0007]

    line_ = ''

    eval_itrs = 50
    bc_tested = False
    bc_epochs = 20
    rule_based_expert = True 
    train_forward = False
    bc_=  True
    use_mappo = False

    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    def make_env(agents,max_cycle,mappo=False):
        if mappo:
            return make_env_s_uni(agents,max_cycle)
        else:
            return make_env_adv_s(agents,max_cycle,num_ma_env=8,seed=42)


    vec_num = 4

    expert = expert_adv(n_agents=n_agents+1)
    #global_expert = expert_2_multi_adv(n_agents=n_agents+1)
    if rule_based_expert:
        env = make_env_adv_s(n_agents,max_cycles)
        ##1 test rule based expert:
        # -150
        #-20.78680337905884 1.932325419984574
        #reward, r_std = evaluate_policy(expert, env, 50)
        e_ade,e_fde,reward, r_std = compare_xx(expert,expert,n_agents=n_agents,env_type=env_name,ma=use_mappo)
        line_ += 'Expert R={0}, STD={1}, ADE={2}, FDE={3} \n'.format(
            np.mean(reward),r_std,e_ade,e_fde)
        print(line_)

    ##2 test nn expert (MAPPO) and train it
    elif train_forward:
        # in case not mappo use comments
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                        net_arch=[64,dict(pi=[64, 32], vf=[64, 32])])
        class policy_mappo(MA_ActorCritic):
            def __init__(self, observation_space, action_space, lr_schedule, n_agents=n_agents, **kwargs):
                super().__init__(observation_space, action_space, lr_schedule, n_agents, **kwargs)

        venv_single_mappo = make_env(n_agents,max_cycles,mappo=use_mappo)

        expert = PPO(
            policy=policy_mappo,#MlpPolicy,
            env=venv_single_mappo,
            batch_size=256,
            ent_coef=0.0,
            learning_rate=0.0007,
            n_epochs=10,
            n_steps=1024,
            tensorboard_log = './gail_log',
            #seed=SEED,
            #policy_kwargs = policy_kwargs,
            device='cuda',
        )
        env = make_env(n_agents,max_cycles,mappo=use_mappo)# make_env_single()#f1()#
        reward, r_std = evaluate_policy(expert, env, 50)
        print('Before:  ',reward,r_std)
        for _ in range(1):
            expert.learn(15e6)  # Note: set to 100000 to train a proficient expert
            reward, r_std = evaluate_policy(expert, env, 50)
            print(reward,r_std)
            breakpoint()


    ##3 generate experts trajs

    global_expert = expert_2_multi_adv(n_agents=n_agents+1)
    ## collective
    try:
        rollouts_coop = types.load(f'rollouts_coop_adv_{num_of_exp_eps}_{n_agents}.npz')
    except:
        #file not saved
        rollouts_coop = rollout.rollout(
            global_expert,
            DummyVecEnv([lambda: RolloutInfoWrapper(grouped_env_adv(SEED,n_agents,max_cycles))] * 8),
            rollout.make_sample_until(min_timesteps=None, min_episodes=num_of_exp_eps),
        )
        types.save(f'rollouts_coop_adv_{num_of_exp_eps}_{n_agents}.npz',rollouts_coop)

    rollouts_single = from_many_to_single(rollouts_coop,n_agents=n_agents+1,ma=use_mappo)
    print('Expert Trajectories Generated')

    venv = make_env(n_agents,max_cycles,mappo=use_mappo)# make_env_vec(seed=10) 
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=[dict(pi=[16, 16], vf=[16, 16])])

    learner_ = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=512,
        ent_coef=0.0,
        learning_rate=lr,#0.003,
        n_epochs=10,
        n_steps=1024,
        tensorboard_log = './gail_single_gen',
        #policy_kwargs = policy_kwargs,
        device='cuda',
    )

    learner_.save('bc_init_s_{}_{}'.format(n_agents,env_name))
    ##4 train BC as warm-up

    if bc_:
        transitions = rollout.flatten_trajectories(rollouts_single)
        from imitation.algorithms import bc

        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=transitions,
            policy=learner_.policy,
            #rng=rng,
        )
        try:
            learner_ = PPO.load('bc_trained_{}_{}_{}'.format(n_agents,num_of_exp_eps,env_name))
            bc_trainer = bc.BC(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                demonstrations=transitions,
                policy=learner_.policy,
                #rng=rng,
           )
        except:
            #env = make_env(n_agents,max_cycles,mappo=use_mappo) #make_env_vec(seed=100)
            #reward_rand, rand_std = evaluate_policy(bc_trainer.policy, env, eval_itrs)
            r_ade,r_fde,reward_rand, rand_std = compare_xx(expert,bc_trainer.policy,n_agents=n_agents,env_type=env_name,ma=use_mappo)
            line_ += 'Random R={0}, STD={1}, ADE={2}, FDE={3} \n'.format(
                reward_rand,rand_std,r_ade,r_fde)

            for s in range(15):
                bc_trainer.train(n_epochs=1)

                #reward_bc, std_bc = evaluate_policy(bc_trainer.policy, env, eval_itrs)
                bc_ade,bc_fde,reward_bc, std_bc = compare_xx(expert,bc_trainer.policy,n_agents=n_agents,env_type=env_name,ma=use_mappo)
                line_ += 'BC({4}/15 epochs) R={0}, STD={1}, ADE={2}, FDE={3} \n'.format(
                    reward_bc,std_bc,bc_ade,bc_fde,s)
                print(line_.splitlines()[-1])

    # NOTE  warm startup:
    learner_.save('bc_trained_{}_{}_{}'.format(n_agents,num_of_exp_eps,env_name))

    if warm_start:
        learner = PPO.load('bc_trained_{}_{}_{}'.format(n_agents,num_of_exp_eps,env_name))
    else:

        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            batch_size=512,
            ent_coef=0.0,
            learning_rate=lr,#0.003,
            n_epochs=10,
            n_steps=1024,
            tensorboard_log = './gail_single_gen',
            #policy_kwargs = policy_kwargs,
            device='cuda',
        )

    #breakpoint()
    ##5 set and train GAIL

    shaped_reward_net = Shaped_R_Blended(venv.observation_space, venv.action_space,n_agents=n_agents+1,lr=0.01)#.cuda()
    shaped_reward_net.init_f_s_a(mode=test_mode)

    airl_trainer = AIRL_ENS(
        demonstrations=rollouts_single,
        demo_batch_size=2**(int(np.log2(num_of_exp_eps*(max_cycles-1)))),
        gen_replay_buffer_capacity=4096,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=shaped_reward_net,
        gen_train_timesteps = 10000,#5000
        init_tensorboard = True,
        init_tensorboard_graph = True,
        disc_opt_kwargs = {'lr':0.007}, #lr for reward
        #debug_use_ground_truth=True,
        n_agents=n_agents+1,
    )

    #env = make_env(n_agents,max_cycles,mappo=use_mappo)#make_env_vec(seed=100) 


    total_timesteps = 500000#200000
    #gail_trainer.train(total_timesteps)  # Note: set to 300000 for better results

    results_file = './single_gail_hot_adv/{0}_single_gail_ds_{1}_agents_{2}_test_{3}.txt'.format(['cold','warm'][warm_start],
        num_of_exp_eps,n_agents,test_mode)

    with open(results_file,'w') as f:
        f.writelines(line_)

    n_rounds = int(total_timesteps // airl_trainer.gen_train_timesteps)
    print('gen steps',airl_trainer.gen_train_timesteps)
    print('dis steps',airl_trainer.n_disc_updates_per_round)
    for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
        print('*** ',r)
        airl_trainer.train_gen(airl_trainer.gen_train_timesteps)
        if False :#r==0:
            #just to avoid forgetting the first time
            bc_trainer.train(n_epochs=1)

        train_batch,r_error = airl_trainer.prepare_for_disc()
        disc_batch = airl_trainer._make_disc_train_batch(**train_batch)

        for _ in range(airl_trainer.n_disc_updates_per_round):



            # STEP 1: train single_reward
            if test_mode==0:
                shaped_reward_net.train_single(disc_batch)
                # STEP2: train group_reward
                shaped_reward_net.train_group(disc_batch)
                #reward_net.group_reward
                with networks.training(airl_trainer.reward_train):
                    # switch to training mode (affects dropout, normalization)

                    airl_trainer.train_disc(**train_batch)
                    #gail_trainer.train_disc()
            elif test_mode==1:
                shaped_reward_net.train_single(disc_batch)
            elif test_mode==2:
                shaped_reward_net.train_group(disc_batch)

            # DONE!

        airl_trainer.logger.dump(airl_trainer._global_step)
        print(f'reward error{r_error}')

        if r%5==0 or r%(n_rounds-1)==0:
            #r_v, r_std= evaluate_policy(
            #    learner, env, eval_itrs, return_episode_rewards=False
            #    )
            l_ade, l_fde,r_v, r_std = compare_xx(expert,learner,n_agents=n_agents,env_type=env_name,ma=use_mappo)
            line = '{5} GAIL step {4}: R={0}, STD={1}, ADE={2}, FDE={3} \n'.format(
                r_v,r_std,l_ade,l_fde,r,['cold','warm'][warm_start])

            with open(results_file,'a') as f:
                f.writelines(line) 
            print(line)

    ##save
    import datetime
    now_ = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    learner.save('trainer_{}_test_{}_{}'.format(num_of_exp_eps,test_mode,env_name))
    th.save(shaped_reward_net.state_dict(), 'ens_rewards_{}_test_{}_{}'.format(num_of_exp_eps,test_mode,env_name))
    del venv
    gc.collect()

if __name__ == '__main__':

    for ds_size in [25,45]:#10
        for test_mode in [0,1,2]:
            if (ds_size == 25) and test_mode in [0,1]:
                continue
            main(num_of_exp_eps=ds_size,test_mode=test_mode)
            gc.collect()