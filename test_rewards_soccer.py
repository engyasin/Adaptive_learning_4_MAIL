
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import  PPO

from multi_ppo import ppo_2_multi, unified_Mlp , MA_ActorCritic, MA_Disc,ppo_compete_ma
from make_envs import grouped_env,make_env_s,make_env_s_uni,make_env_adv_s
from tests_utils import save_frames_as_gif

from imitation.data import rollout
from expert import expert_p, expert_2_multi
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from ensemble import from_many_to_single
from torch.nn import functional as F

from mixed_spread_main import ens_rews_full,fix_coop
import numpy as np
import cv2

from envs_utils import unified_env,VecExtractDictObs,Vecbooldone_Soccer,unified_env_soccer
from gym.envs.registration import register
from petting_zoo_v import parallel_soccer
import gym
import supersuit as ss

import time
import torch as th
from ensemble import from_many_to_single,  DsEnsemble
from ensemble import from_many_to_single, convert_batch, DsEnsemble4,Shaped_R_Blended

n_agents = 3
max_cycles = 25
num_of_exp_eps = 400
save_video = True
SEED = 42
font = cv2.FONT_HERSHEY_SIMPLEX

register(
    id='multigrid-soccer-v0',
    entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
)

num_cpus = 1
def make_env_s(n_agents,max_cycles,flatt_obs=True,num_ma_env=4,seed=42):
    env = parallel_soccer(lambda: gym.make('multigrid-soccer-v0'),n_agents=n_agents,  max_cycles=max_cycles)
    env.reset(seed=seed)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env)

    return Vecbooldone_Soccer(n_agents,ss.concat_vec_envs_v1(env_, num_ma_env, num_cpus=num_cpus, base_class='stable_baselines3'),
                              flatt_obs=flatt_obs)

"""
learner_single = PPO(
    policy=policy_mappo,
    env=venv_single_mappo,
    batch_size=512,
    ent_coef=0.0,
    learning_rate=0.0007,
    n_epochs=10,
    n_steps=1024,
    tensorboard_log = './gail_log',
    #seed=SEED,
    device='cuda',
)
"""

def one_hot(arr,size):
    b = np.zeros((arr.size, size),dtype=np.int64)
    b[np.arange(arr.size), arr] = 1
    return b
def agt_pos(ps,i):
    p = ps[i]
    cam_range = np.max(np.abs(np.array(ps)))
    x, y = p
    y *= (
        -1
    )  # this makes the display mimic the old pyglet setup (ie. flips image)
    x = (
        (x / cam_range) * 700 // 2 * 0.9
    )  # the .9 is just to keep entities from appearing "too" out-of-bounds
    y = (y / cam_range) * 700 // 2 * 0.9
    x += 700 // 2
    y += 700 // 2
    return int(x)-8,int(y)-8

names_dict_sp = {10:['29-04-2023_14-49-44','29-04-2023_15-33-30','29-04-2023_16-20-01'],
                 20:['29-04-2023_17-13-52','29-04-2023_18-03-46','29-04-2023_18-44-59'],
                 40:['29-04-2023_19-40-46','29-04-2023_20-31-17','29-04-2023_21-24-25'],}
#14-49-44
def make_env(agents,max_cycle,env_name='spread'):
    return make_env_s(n_agents=agents,max_cycles=max_cycle)#,flatt_obs=False)


def main(test_mode=0,env_name='soccer',ds=10):
    n_agents,max_cycles = 4,15
    # load learner

    then  = names_dict_sp[ds][test_mode]
    #then = ds
    learner_single = PPO.load('trainer_{}_{}_test_{}'.format(then,env_name,test_mode))

    # create envs

    venv = make_env(n_agents,max_cycles,env_name=env_name)# make_env_vec(seed=10) 

    #load rewards
    reward_net = Shaped_R_Blended(venv.observation_space, venv.action_space,n_agents=n_agents+(env_name=='adversary'),lr=0.005)
    reward_net.init_f_s_a(mode=test_mode)
    reward_net.load_state_dict(th.load('ens_rewards_{}_{}_test_{}'.format(
        env_name,then,test_mode),map_location=th.device('cpu'))
        )


    #breakpoint()

    # loop over
    frames = []
    state = venv.reset()
    scale = 1e4
    for i in range(max_cycles*15):
        actions ,_  = learner_single.predict(state)
        #actions ,_  = bc_policy.policy.predict(state)
        n_state,rewards ,dones ,info= venv.step(actions)
        with th.no_grad():
            state_t = th.Tensor(state)
            actions_t = th.Tensor(one_hot(actions,8))#np.expand_dims(actions,1))
            n_state_t = th.Tensor(n_state)
            dones_t = np.expand_dims(dones,1)

            reward_net.group_trained = 1
            reward_net.single_trained = 0
            r_c = reward_net.new_reward(state,actions,n_state,dones)*scale

            reward_net.group_trained = 0
            reward_net.single_trained = 1
            r_s = reward_net.new_reward(state,actions,n_state,dones)*scale

            reward_net.group_trained = 1
            r_e = reward_net.new_reward(state,actions,n_state,dones)*scale

            full_states = th.hstack((state_t.detach().clone().reshape(-1,np.prod(reward_net.group_obs.shape)),
                        actions_t.detach().clone().reshape(-1,reward_net.n_agents*reward_net.action_space.n)))
            r_a = reward_net.f_s_a(full_states)*n_agents#24,3
            r_a = 1-r_a.detach().numpy()[0]


        # draw all
        state = n_state
        venv.render()

        print(r_c)
        print(r_s)
        print(r_e)
        print(r_a)
        print('*********')
        #if (r_a<(0.98)).any():
        #    breakpoint()

        frame = venv.render(mode="rgb_array")
        #breakpoint()
        #for m in range(0):#n_agents):
        #    cv2.putText(frame,str(m+1),
        #        agt_pos(n_state[0,:,2:4],m), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.copyMakeBorder(frame,250,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        line = '     '.join(['--',f'r_s*{scale:.0e}','h_s_a',f'r_h*{scale:.0e}'])
        #line = '   '.join(['--','--','r_att','r_ens'])
        line_width = 28
        font_scale = 0.7

        cv2.putText(frame,line,
                (0,line_width), font, font_scale-0.1, (0, 0, 0),2 , cv2.LINE_AA)
        #breakpoint()
        #line = 'r_c   {0:.2f}   {1:.2f}  -'.format(r_c[0],r_a[-1])
        line = 'r_c     {0:.2f}      *'.format(r_c[0])

        cv2.putText(frame,line,
                (0,2*line_width), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        for n in range(n_agents):
            line = 'r_s{3}     {0:.2f}     {1:.2f}     {2:.2f}'.format(r_s[n],r_a[n],r_e[n],n+1)
            if n<(n_agents/2):
                # green team
                color = (0,255,0)
            else:
                #blue team
                color = (0,0,255)
            cv2.putText(frame,line,
                    (0,(3+n)*line_width), font, font_scale, color, 2, cv2.LINE_AA)
        #cv2.putText(frame,'r_s {0:.2f}, r_c {1:.2f} , r_e {2:.2f}'.format(r_s[0],r_c[0],r_e[0]),
        #        (0,50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        frames.append(frame)
        #time.sleep(0.1)
        if (any((r_a).astype(int) != r_a)) or any(r_e>0.01):
            f_ = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imwrite(env_name+f'_snaps/{len(frames)}_.jpg',f_)

    if save_video: 
        save_frames_as_gif(frames,filename='{}_gail_trained_with_fsa_{}_{}.gif'.format(env_name,n_agents,
            num_of_exp_eps))
    # save





if __name__=='__main__':
    #10,20,40
    main(ds=10,env_name='soccer')