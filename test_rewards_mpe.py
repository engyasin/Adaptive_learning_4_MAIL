
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



class policy_mappo(MA_ActorCritic):

    def __init__(self, observation_space, action_space, lr_schedule, n_agents=n_agents, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, n_agents,
            net_arch = [dict(pi=[128, 64], vf=[64, 64])],**kwargs)

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

names_dict_sp = {10:['24-04-2023_14-04-10','24-04-2023_15-58-02','24-04-2023_17-42-54'],
                 #20:['24-04-2023_19-42-32','24-04-2023_21-29-18','24-04-2023_23-14-28'],
                 #40:['25-04-2023_01-14-25','25-04-2023_03-01-52','25-04-2023_04-47-37'],
                 80:['25-04-2023_08-20-24','25-04-2023_11-21-12','25-04-2023_14-21-10'],
                 160:['26-04-2023_03-32-19','26-04-2023_06-32-48','26-04-2023_09-32-13'],}

names_dict_adv = {#10:['30-04-2023_14-04-10','24-04-2023_15-58-02','24-04-2023_17-42-54'],
                 #40:['30-04-2023_19-42-32','24-04-2023_21-29-18','24-04-2023_23-14-28'],
                 80:['30-04-2023_17-29-34','30-04-2023_17-49-21','30-04-2023_18-08-39'],
                 160:['30-04-2023_18-30-41','30-04-2023_18-50-31','30-04-2023_19-10-17'],}
def make_env(agents,max_cycle,env_name='spread'):
    if env_name=='spread':
        return make_env_s(agents,max_cycle,num_ma_env=1,seed=42)
    else:
        return make_env_adv_s(agents,max_cycle,num_ma_env=8,seed=42)

def main(test_mode=0,env_name='spread',ds=10):
    n_agents,max_cycles = 3,15
    # load learner
    if env_name != 'spread':
        then  = names_dict_adv[ds][test_mode]
        #then = ds
        learner_single = PPO.load('trainer_{}_test_{}_{}'.format(then,test_mode,env_name))
    else:
        then  = names_dict_sp[ds][test_mode]
        learner_single = PPO.load('trainer_{}_test_{}'.format(then,test_mode))

    # create envs

    venv = make_env(n_agents,max_cycles,env_name=env_name)# make_env_vec(seed=10) 

    #load rewards
    reward_net = Shaped_R_Blended(venv.observation_space, venv.action_space,n_agents=n_agents+(env_name=='adversary'),lr=0.005)
    reward_net.init_f_s_a(mode=test_mode)
    if env_name != 'spread':
        reward_net.load_state_dict(th.load('ens_rewards_{}_test_{}_{}'.format(
            then,test_mode,env_name),map_location=th.device('cpu'))
            )
    else:
        reward_net.load_state_dict(th.load('ens_rewards_{}_test_{}'.format(
            then,test_mode),map_location=th.device('cpu'))
            )



    # loop over
    frames = []
    state = venv.reset()
    scale = 1e6
    for i in range(max_cycles*15):
        actions ,_  = learner_single.predict(state)
        #actions ,_  = bc_policy.policy.predict(state)
        n_state,rewards ,dones ,info= venv.step(actions)
        with th.no_grad():
            state_t = th.Tensor(state)
            actions_t = th.Tensor(one_hot(actions,5))#np.expand_dims(actions,1))
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
            r_a = 1-(r_a.detach().numpy()[0])


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
        #for m in range(0):#n_agents):
        #    cv2.putText(frame,str(m+1),
        #        agt_pos(n_state[0,:,2:4],m), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.copyMakeBorder(frame,200,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))

        line = '     '.join(['--',f'r_s*{scale:.0e}','h_s_a',f'r_h*{scale:.0e}'])
        line_width = 35

        font_scale = 0.8

        cv2.putText(frame,line,
                (0,line_width), font, font_scale-0.1, (0, 0, 255), 2, cv2.LINE_AA)
        #breakpoint()
        #line = 'r_c   {0:.2f}   {1:.2f}  -'.format(r_c[0],r_a[-1])
        line = 'r_c     {0:.2f}      *'.format(r_c[0])

        cv2.putText(frame,line,
                (0,2*line_width), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
        for n in range(n_agents+(env_name=='adversary')):
            line = 'r_s{3}     {0:.2f}     {1:.2f}     {2:.2f}'.format(r_s[n],r_a[n],r_e[n],n+1)

            cv2.putText(frame,line,
                    (0,(3+n)*line_width), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
        #cv2.putText(frame,'r_s {0:.2f}, r_c {1:.2f} , r_e {2:.2f}'.format(r_s[0],r_c[0],r_e[0]),
        #        (0,50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        frames.append(frame)

        if not(all(r_a == r_a[0])):
            f_ = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imwrite(env_name+f'_snaps/{len(frames)}_.jpg',f_)

    if save_video: 
        save_frames_as_gif(frames,filename='{}_gail_trained_with_fsa_{}_{}.gif'.format(env_name,n_agents,
            num_of_exp_eps))
    # save





if __name__=='__main__':
    main(ds=80,env_name='adversary')
    #main(ds=10,env_name='spread')