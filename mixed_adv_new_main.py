#from pettingzoo.mpe import simple_reference_v2
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss
import torch as th
#import seals
import tqdm
from gym import spaces
from imitation.algorithms.adversarial.gail import GAIL
from torch.nn import functional as F

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards import reward_nets, reward_wrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import networks
from imitation.util.networks import RunningNorm
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import MlpPolicy

from compare_policies import compare_m_actions, compare_x
from ensemble import DsEnsemble, convert_batch, from_many_to_single
from envs_utils import VecExtractDictObs, unified_env
from expert import expert_2_multi, expert_2_multi_adv, expert_adv, expert_p
from multi_ppo import (MA_ActorCritic, MA_Disc, ppo_2_multi, ppo_compete_ma,
                       unified_Mlp)

################################
#warppers
################################



def fix_comp(f3_comp,n_agents= 4,ma=False):


    def combined_fs( state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,):


        if ma:
            #NOTE this is for the default order (change start to get different order)
            state_c = state.reshape(-1,state.shape[-1]*n_agents)[::n_agents]
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)[::n_agents]
        else:
            state_c = state.reshape(-1,state.shape[-1]*n_agents)
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)

        action_c = action.reshape(action.shape[0]//n_agents,-1)
        done_c = done.reshape(-1,n_agents).astype(bool)

        #breakpoint()

        r_comp = f3_comp(state_c,
                action_c,next_state_c,
                done_c, **kwargs)

        r_comp = np.array([r_comp]*n_agents).T.flatten()


        return r_comp

    return combined_fs

def fix_coop(f2_coop,n_agents= 4,ma=False):


    def combined_fs( state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,):


        if ma:
            #NOTE this is for the default order (change start to get different order)
            state_c = state.reshape(-1,state.shape[-1]*n_agents)[::n_agents]
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)[::n_agents]
        else:
            state_c = state.reshape(-1,state.shape[-1]*n_agents)
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)

        action_c = action.reshape(action.shape[0]//n_agents,-1)
        done_c = done.reshape(-1,n_agents).astype(bool)

        #breakpoint()

        r_coop = f2_coop(state_c,
                action_c,next_state_c,
                done_c, **kwargs)


        r_coop = np.array([r_coop]*n_agents).T.flatten()

        # ensemble:

        #r_final = r_single + (r_coop/n_agents)

        # between 0-1
        #r_final = r_final/(1+1/n_agents)

        #breakpoint()
        return r_coop

    return combined_fs


def ensemble_rews_dummy(f1_single,f2_coop,n_agents= 4,ma=False):
    #combine two rewards functions
    # single: is ok as is
    # coop: need to reshap input and output
    # according to the agents number

    def combined_fs( state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,):

        r_single = f1_single(state,
                action,next_state,
                done, **kwargs)

        if ma:
            #NOTE this is for the default order (change start to get different order)
            state_c = state.reshape(-1,state.shape[-1]*n_agents)[::n_agents]
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)[::n_agents]
        else:
            state_c = state.reshape(-1,state.shape[-1]*n_agents)
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)

        action_c = action.reshape(action.shape[0]//n_agents,-1)
        done_c = done.reshape(-1,n_agents).astype(bool)

        #breakpoint()

        r_coop = f2_coop(state_c,
                action_c,next_state_c,
                done_c, **kwargs)


        if False:
            #should we N root them?
            r_coop = r_coop ** (1/n_agents)
            r_comp = r_comp ** (1/(n_agents-1))

        r_coop = np.array([r_coop]*n_agents).T.flatten()



        # ensemble:
        #r_final = (r_coop**(1.0/n_agents))*r_single*(r_comp**(1./(1-n_agents)))

        #r_final = r_single + (r_coop/n_agents)

        # between 0-1
        #r_final = r_final/(1+1/n_agents)

        #breakpoint()
        return r_single

    return combined_fs

def ens_rews_full(f1_single,f2_coop,EnsNet,n_agents= 4,ma=False,show_attnd=False):
    #combine two rewards functions
    # single: is ok as is
    # coop: need to reshap input and output
    # according to the agents number

    def combined_fs( state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,):

        # if ma will take the first only
        r_single = f1_single(state,
                action,next_state,
                done, **kwargs)

        if ma:
            #NOTE this is for the default order (change start to get different order)
            state_c = state.reshape(-1,state.shape[-1]*n_agents)[::n_agents]
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)[::n_agents]
        else:
            state_c = state.reshape(-1,state.shape[-1]*n_agents)
            next_state_c = next_state.reshape(-1,next_state.shape[-1]*n_agents)

        action_c = action.reshape(action.shape[0]//n_agents,-1)

        done_c = done.reshape(-1,n_agents).astype(bool)

        #breakpoint()

        r_coop = f2_coop(state_c,
                action_c,next_state_c,
                done_c, **kwargs)


        if False:
            #should we N root them?
            r_coop = r_coop ** (1/n_agents)
            r_comp = r_comp ** (1/(n_agents-1))


        r_single = r_single.reshape(r_single.shape[0]//n_agents,-1)

        inputs = np.hstack((r_single,r_coop.reshape(-1,1)))

        #r_coop = np.array([r_coop]*n_agents).T.flatten()

        # ensemble:
        r_final = EnsNet(th.from_numpy(inputs).to('cuda')).flatten()

        if show_attnd:
            att = EnsNet.attend(th.from_numpy(inputs).to('cuda'))

            with open('adv_att_w_{}_agnt_{}.txt'.format(ma,n_agents),mode='a') as f:
                f.write('\n')
                f.write(str(att))

        #r_final = r_single + (r_coop/n_agents)

        # between 0-1
        #r_final = r_final/(1+1/n_agents)

        #breakpoint()
        return r_final.detach().cpu().numpy()

    return combined_fs



##########################################
######## TODO  ###########################
##########################################


def main(n_agents=3,num_of_exp_eps=50, mixed=True):

    import random

    SEED = 42
    local_r = 0.5
    debug = False
    ma = True
    max_cycles = 25
    bc_tested = False
    bc_epochs = 20

    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    np.set_printoptions(precision=3)

    from make_envs import grouped_env_adv, make_env_adv_s, make_env_adv_s_uni

    class unified_Mlp_inst(unified_Mlp):
        """
        Grouped Mappo
        """
        def __init__(self, observation_space, action_space, lr_schedule, n_agents=n_agents, **kwargs):
            super().__init__(observation_space, action_space, lr_schedule, n_agents+1, **kwargs)

    def policy_mappo(observation_space,action_space,lr_schedule,n_agents=n_agents, **kwargs):
        """
        single case
        """
        return MA_ActorCritic(observation_space,action_space,lr_schedule,n_agents=n_agents+1, 
                   net_arch = [dict(pi=[128, 64], vf=[64, 64])], **kwargs)


    ##STEP 1: generate experts trajs
    global_expert = expert_2_multi_adv(n_agents=n_agents+1)
    ## collective
    rollouts_coop = rollout.rollout(
        global_expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(grouped_env_adv(SEED,n_agents,max_cycles))] * 8),
        rollout.make_sample_until(min_timesteps=None, min_episodes=num_of_exp_eps),
    )

    #breakpoint()
    ##decentralized
    if debug:
        local_expert = expert_p(n_agents=n_agents)
        expert_rewards, _ = evaluate_policy(
            global_expert,
            DummyVecEnv([lambda: RolloutInfoWrapper(grouped_env_adv(SEED,n_agents,max_cycles))] * 8),
            100, return_episode_rewards=True
        )

        print('single expert reward: ',np.mean(expert_rewards))


    rollouts_single = from_many_to_single(rollouts_coop,n_agents=n_agents+1,ma=ma)

    venv_single = make_env_adv_s(n_agents,max_cycles,num_ma_env=1,seed=SEED)
    # won't be trained
    venv_coop_s = make_env_adv_s(n_agents,max_cycles,num_ma_env=1,seed=SEED)

    venv_coop_m = DummyVecEnv([lambda :grouped_env_adv(SEED,n_agents,max_cycles)] * (n_agents+1))

    if ma:
        venv_single_mappo = make_env_adv_s_uni(n_agents,max_cycles)

        learner_single = PPO(
            policy=policy_mappo,
            env=venv_single_mappo,
            batch_size=256,
            ent_coef=0.0,
            learning_rate=0.0007,
            n_epochs=10,
            n_steps=1024,
            tensorboard_log = './gail_log',
            #seed=SEED,
            device='cuda',
        )

        reward_net_single = MA_Disc(
            venv_single_mappo.observation_space, venv_single_mappo.action_space, normalize_input_layer=RunningNorm
        )

        transitions = rollout.flatten_trajectories(rollouts_single)
        from imitation.algorithms import bc
        from imitation.util import logger as imit_logger

        #format_strs = 'log'
        log_obj = imit_logger.configure(format_strs=['log'])

        bc_trainer = bc.BC(
            observation_space=venv_single_mappo.observation_space,
            action_space=venv_single_mappo.action_space,
            demonstrations=transitions,
            policy = MA_ActorCritic(venv_single_mappo.observation_space,
                                    venv_single_mappo.action_space,
                                    lr_schedule=lambda _: th.finfo(th.float32).max,
                                    n_agents=n_agents+1),
            device = "cuda",
            custom_logger = log_obj,
            #rng=rng,
        )
        bc_trainer.train(n_epochs=1)

        single_gen = PPO(
            policy=policy_mappo,
            env=venv_single_mappo,
            batch_size=256,
            ent_coef=0.0,
            learning_rate=0.0007,
            n_epochs=10,
            n_steps=1024,
            device='cuda',
            #seed=SEED,
            #tensorboard_log = './gail_log',
        )

        #single_gen.set_parameters(bc_trainer.get_parameters())

        coop_gen = PPO(
            policy=policy_mappo,
            env=venv_single_mappo,
            batch_size=512,
            ent_coef=0.0,
            learning_rate=0.0007,
            n_epochs=10,
            n_steps=1024,
            device='cuda',
            #seed=SEED,
            #tensorboard_log = './gail_log',
        )
        #coop_gen.set_parameters(bc_trainer.get_parameters())

    else:

        learner_single = PPO(
            policy=MlpPolicy,
            env=venv_single,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
            #seed=SEED,
            tensorboard_log = './gail_log',
        )
        reward_net_single = BasicRewardNet(
            venv_single.observation_space, venv_single.action_space, normalize_input_layer=RunningNorm
        )

        single_gen = PPO(
            policy=MlpPolicy,
            env=venv_single,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
            #seed=SEED,
            tensorboard_log = './gail_log',
        )

        coop_gen = PPO(
            policy=MlpPolicy,
            env=venv_single,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
            #seed=SEED,
            tensorboard_log = './gail_log',
        )

    try:
        pass
        #if ma:
        #    learner_single = PPO.load('learner_init_{}'.format(n_agents),env=venv_single_mappo)
        #    single_gen  = PPO.load('learner_init_{}'.format(n_agents),env=venv_single_mappo)
        #    comp_gen= PPO.load('learner_init_{}'.format(n_agents),env=venv_single_mappo)
        #    coop_gen= PPO.load('learner_init_{}'.format(n_agents),env=venv_single_mappo)
        #else:
        #    learner_single = PPO.load('learner_init_{}'.format(n_agents),env=venv_single)
        #    single_gen = PPO.load('learner_init_{}'.format(n_agents),env=venv_single)
        #    comp_gen = PPO.load('learner_init_{}'.format(n_agents),env=venv_single)
        #    coop_gen = PPO.load('learner_init_{}'.format(n_agents),env=venv_single)
    except :
        learner_single.save('learner_init{}'.format(n_agents))


    # TODO learner coop and single, same policy net, not hard set
    # it will not be trained
    policy_kwargs = dict(net_arch=[dict(pi=[128, 64], vf=[64, 64])])
    learner_coop = ppo_2_multi(
        n_agents=n_agents+1,
        env=venv_coop_s,
        policy=unified_Mlp_inst,
        batch_size=512,
        ent_coef=0.0,
        learning_rate=0.0007,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=policy_kwargs,
        )


    reward_net_coop = BasicRewardNet(
        venv_coop_m.observation_space, venv_coop_m.action_space, normalize_input_layer=RunningNorm
    )

    EnsRew = DsEnsemble(n_agents=n_agents+1,lr=0.05).cuda()

    n_disc_updates_per_round = 8

    #breakpoint()
    # output folder for TFB for dicremnator
    gail_trainer_coop = GAIL(
        demonstrations=rollouts_coop,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv_coop_m,
        gen_algo=learner_coop,
        reward_net=reward_net_coop,
        #allow_variable_horizon=True,
        init_tensorboard = True,
        init_tensorboard_graph =True,
    )

    if ma:

        gail_trainer_single = GAIL(
            demonstrations=rollouts_single,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=n_disc_updates_per_round,
            venv=venv_single_mappo,
            gen_algo=learner_single,
            reward_net=reward_net_single,
            #allow_variable_horizon=True,
            init_tensorboard = True,
            init_tensorboard_graph =True,
        )

    else:
        gail_trainer_single = GAIL(
            demonstrations=rollouts_single,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=n_disc_updates_per_round,
            venv=venv_single,
            gen_algo=learner_single,
            reward_net=reward_net_single,
            #allow_variable_horizon=True,
            init_tensorboard = True,
            init_tensorboard_graph =True,
        )


    #learner_rewards_before_training, _ = evaluate_policy(
    #    learner_single, venv_single, 100, return_episode_rewards=True
    #)

    time_steps_train = 35e6#6
    gen_train_timesteps = 5e4#4

    n_rounds = int(time_steps_train//gen_train_timesteps)


    for r in tqdm.tqdm(range(0, n_rounds), desc="round"):

        gail_trainer_single.train_gen(total_timesteps=gen_train_timesteps)

        gail_trainer_coop.gen_algo.policy.set_actor_parameters(
            gail_trainer_single.gen_algo.policy.get_actor_parameters()
        )


        gail_trainer_single.gen_callback = gail_trainer_single.venv_wrapped.make_log_callback()
        gail_trainer_single.gen_algo.set_env(gail_trainer_single.venv_wrapped)

        # just to generate trajs
        learner_coop.simulte_trajs(gail_trainer_coop,total_timesteps=int(gen_train_timesteps//2))

        for _ in range(n_disc_updates_per_round):
            with networks.training(gail_trainer_single.reward_train):
                # switch to training mode (affects dropout, normalization)

                gail_trainer_single.train_disc()

            with networks.training(gail_trainer_coop.reward_train):
                gail_trainer_coop.train_disc()

            # TODO ensemble
            if mixed:
                ## add modified reward to gen env
                ## change callback also
                batch = gail_trainer_coop._make_disc_train_batch()
                Dt = gail_trainer_coop.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                ).to('cuda')
                Dt = -F.logsigmoid(-Dt)

                n_batch = convert_batch(batch,n_agents=n_agents+1,ma=ma)
                #breakpoint()

                Di = gail_trainer_single.logits_expert_is_high(
                    n_batch["state"],
                    n_batch["action"],
                    n_batch["next_state"],
                    n_batch["done"],
                ).to('cuda')
                Di = -F.logsigmoid(-Di)


                # train
                EnsRew.train()
                #new_lr = (r/n_rounds)*old_lr
                #new_lr = 0.005#(10**((r/n_rounds)))/100
                #print('new lr: ',new_lr)
                EnsRew.train_(Dt,Di,n_batch['labels_expert_is_one'].to('cuda'),
                        lr=0.01)

                EnsRew.eval()

        gail_trainer_single.logger.dump(gail_trainer_single._global_step)

        #new_rew = ensemble_rews(gail_trainer_single.reward_train.predict_processed,
        #    gail_trainer_coop.reward_train.predict_processed,n_agents= n_agents)
        if mixed:
            new_rew = ens_rews_full(gail_trainer_single.reward_train.predict_processed,
                gail_trainer_coop.reward_train.predict_processed,
                EnsRew.eval(),n_agents= n_agents+1,
                ma=ma)
        else:
            new_rew = ensemble_rews_dummy(gail_trainer_single.reward_train.predict_processed,
                gail_trainer_coop.reward_train.predict_processed,
                n_agents = n_agents+1,ma=ma)


        venv_ = reward_wrapper.RewardVecEnvWrapper(
            make_env_adv_s_uni(n_agents,max_cycles),
            reward_fn=gail_trainer_single.reward_train.predict_processed,
        )

        single_gen.set_env(venv_)
        single_gen.learn(total_timesteps=gen_train_timesteps)



        rew_coop = fix_coop(gail_trainer_coop.reward_train.predict_processed,
            n_agents= n_agents+1,ma=ma)

        venv_ = reward_wrapper.RewardVecEnvWrapper(
            make_env_adv_s_uni(n_agents,max_cycles),
            reward_fn=rew_coop,
        )
        coop_gen.set_env(venv_)
        coop_gen.learn(total_timesteps=gen_train_timesteps)

        #breakpoint()
        venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
            gail_trainer_single.venv_buffering,
            reward_fn=new_rew,
        )

        gail_trainer_single.gen_callback = venv_wrapped.make_log_callback()
        gail_trainer_single.gen_algo.set_env(venv_wrapped)

        print(str(r) + ' round done!')

        if False:
            if ma:
                eval_env = make_env_s_uni(n_agents,max_cycles)
            else:
                eval_env = make_env_s(n_agents,max_cycles,num_ma_env=1,seed=SEED)

            learner_c_rewards, _ = evaluate_policy(
                learner_single, eval_env, 100, return_episode_rewards=True
            )

            single_rewards, _ = evaluate_policy(
                single_gen, eval_env, 100, return_episode_rewards=True
            )

            coop_rewards, _ = evaluate_policy(
                coop_gen, eval_env, 100, return_episode_rewards=True
            )


            print('current_r: ',np.mean(learner_c_rewards))

            with open('spread_res_{}.txt'.format(ma),mode='a') as f:
                f.write('\n')
                f.write(str(n_agents)+' agents and '+str(num_of_exp_eps)+' exps ,reward is:'+str(
                    np.mean(learner_c_rewards))+' single r: '+str(
                        np.mean(single_rewards))+' coop r: '+str(
                            np.mean(coop_rewards)) )
        elif (r%3)==0:
            # this should be activated
            # rewards is evaluated with tensorboard eps_rewards

            global_expert = expert_2_multi_adv(n_agents=n_agents+1)
            #compare_N_policy
            policy_list = [learner_single,single_gen,coop_gen]
            if not bc_tested:
                policy_list.append(bc_trainer.policy)
                bc_tested = True
            if bc_epochs:
                bc_trainer.train(n_epochs=1)
                bc_epochs -=1
                bc_tested = False
            matching_scores = compare_m_actions(global_expert,policy_list,
                        make_env_adv_s_uni(n_agents,max_cycles),
                        n_agents=n_agents+1,max_cycles=max_cycles)
            #matching_score = compare_x(global_expert,learner_single,n_agents=n_agents)
            print('percentage of matching [combin, single, coop]: {}%'.format(matching_scores))


            with open('adv_res_acts_{}.txt'.format(ma),mode='a') as f:
                f.write('\n')
                f.write(str(n_agents+1)+' agents and '+str(num_of_exp_eps)+' exps ,matching scores [comb,single,coop] in round {} / {}'.format(
                    r,n_rounds)+' is:\n')
                f.write('\t'.join(map(str,matching_scores)))

    if mixed:
        th.save(EnsRew.state_dict(),'adv_{0}_agents_{1}_exps_ensnet_.pth'.format(
            n_agents+1,num_of_exp_eps
        ))

        learner_single.save('learner_adv_{0}_with_{1}_data_.pth'.format(n_agents+1,
            num_of_exp_eps))

        single_gen.save('single_learner_adv_{0}_with_{1}_data_.pth'.format(n_agents+1,
            num_of_exp_eps))
        coop_gen.save('coop_learner_adv_{0}_with_{1}_data_.pth'.format(n_agents+1,
            num_of_exp_eps))


        th.save(reward_net_single,'adv_{0}_agents_{1}_exps_rew_net_.pth'.format(
            n_agents+1,num_of_exp_eps
        ))

        th.save(reward_net_coop,'adv_{0}_agents_{1}_exps_rew_net_coop_.pth'.format(
            n_agents+1,num_of_exp_eps
        ))
 
        # save demo with attention weights

        #gail_trainer_single.gen_algo
        #venv_wrapped
    if False:
        from test import save_frames_as_gif

        with open('spread_att_w_{}_agnt_{}.txt'.format(ma,n_agents),mode='a') as f:
            f.write('\n')
            f.write('*************************')
            f.write('N demos: {}'.format(num_of_exp_eps))

        new_rew = ens_rews_2(gail_trainer_single.reward_train.predict_processed,
            gail_trainer_coop.reward_train.predict_processed,gail_trainer_compete.reward_train.predict_processed,
            EnsRew.eval(),n_agents= n_agents,
            ma=ma,show_attnd=True)

        venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
            gail_trainer_single.venv_buffering,
            reward_fn=new_rew,
        )

        obs = venv_wrapped.reset()
        frames = []
        for i in range(50):

            actions,_ =gail_trainer_single.gen_algo.predict(obs,deterministic=True)
            #[env.action_space.sample() for _ in range(env.num_envs)]
            #ob,re,do,inf = env.step()
            #breakpoint()
            obs ,rewards ,dones ,infos = venv_wrapped.step(actions)
        
            #venv_wrapped.render()
            frames.append(venv_wrapped.render(mode="rgb_array"))
            #time.sleep(0.1)

        #breakpoint()


        save_frames_as_gif(frames,
            filename='spread_agnt_{}_demos_{}_ma_{}_mixed_{}.gif'.format(
                n_agents,num_of_exp_eps,ma,mixed
            ))

    #print('learner_rewards_after_training: ',np.mean(learner_c_rewards))
    #print('learner_rewards_before_training: ',np.mean(learner_rewards_before_training))


    #plt.hist(
    #    [learner_rewards_before_training, learner_rewards_after_training,expert_rewards],
    #    label=["untrained", "trained", "expert"],
    #)
    #plt.legend()
    #plt.title('Combined Reward')
    #plt.show()


if __name__ == '__main__':
    for n_agnt in [3]:
        for exprt in [500]:
            main(n_agents=n_agnt,num_of_exp_eps=exprt,mixed=True)










