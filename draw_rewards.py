import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pettingzoo.mpe import simple_spread_v2,simple_adversary_v2

import supersuit as ss
from stable_baselines3.common.monitor import Monitor

#from make_envs import unified_env, Vec_bool_done, VecRolloutReward
from make_envs import grouped_env,make_env_s,make_env_s_uni

from tests_utils import save_frames_as_gif

import numpy as np
from compare_policies import compare_x,compare_xx

from stable_baselines3.common.evaluation import evaluate_policy

from expert import expert_p,expert_2_multi
from ensemble import from_many_to_single, convert_batch, DsEnsemble2

from matplotlib import pyplot as plt

import cv2
import time


def plot_rs(r_c,r_1,frames,ds=16,test_mode=0,env_name='spread'):
    
    fig = plt.figure(figsize=(12,4))
    crtical = []
    for i,r_r_c in enumerate(r_c):
        plt.subplot(r_c.shape[0],3,(i*3+1))
        plt.plot(np.arange(len(r_r_c)),r_r_c,'-',color='green',label=f'effect_group_ep_{i}')
        plt.plot(np.arange(len(r_r_c)),r_1[i],'-',color='blue',label=f'effect_agent_ep_{i}')
        crtical.extend([np.argmin((r_r_c-r_1[i]))+(i*len(r_r_c)),np.argmax((r_r_c-r_1[i]))+(i*len(r_r_c))])
        plt.plot([np.argmin((r_r_c-r_1[i])),np.argmin((r_r_c-r_1[i]))],[0,max(r_r_c)],'-',color='black',linewidth=2,label='minimum difference')
        plt.plot([np.argmax((r_r_c-r_1[i])),np.argmax((r_r_c-r_1[i]))],[0,max(r_r_c)],'-',color='gray',linewidth=2,label='maximum difference')
        plt.xlabel('Steps')
        plt.ylabel('Attention Factor')
        plt.legend()
        plt.grid()

        plt.subplot(r_c.shape[0],3,(i*3+2))
        plt.imshow(frames[crtical[-2]])
        if i==(len(r_c)-1):
            #last line
            plt.xlabel('High attention to decentralized reward (competation)')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(r_c.shape[0],3,(i*3+3))
        plt.imshow(frames[crtical[-1]])
        if i==(len(r_c)-1):
            #last line
            plt.xlabel('High attention to centralized reward (cooperation)')
        plt.xticks([])
        plt.yticks([])

    fig.tight_layout()#pad=0.5)
    #print(crtical)
    plt.savefig(f'{env_name}_rewards_{ds}_{test_mode}_w.pdf')
    plt.show()

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

def main():
    env_name = 'spread'
    ds_size = 1024
    test_mode = 0

    max_cycles = 25
    n_agents = 3

    save_video = True

    env = make_env_s(n_agents=n_agents,max_cycles=max_cycles,num_ma_env=1,seed=42)

    frames = []
    imgs = []

    #the start + 9
    if ds_size==16:
        model_paths = ["15-01-2023_13-58-52","15-01-2023_17-51-15","15-01-2023_21-44-43"]
    elif ds_size==128:
        model_paths = ["16-01-2023_01-38-08","16-01-2023_05-29-31","16-01-2023_09-18-16"]
    elif ds_size==1024:
        model_paths = ["16-01-2023_13-11-41","16-01-2023_17-01-05","16-01-2023_20-52-22"]


    back_then = model_paths[test_mode]

    learner = PPO.load('trainer_{}_test_{}'.format(back_then,test_mode))
    reward_net = DsEnsemble2(env.observation_space, env.action_space,n_agents=n_agents,lr=0.01)
    state_dict = th.load('ens_rewards_{}_test_{}'.format(back_then,test_mode),map_location=th.device('cpu'))
    reward_net.load_state_dict(state_dict)

    font = cv2.FONT_HERSHEY_SIMPLEX

    state = env.reset()
    r_c_1,r_1_1 = [],[]
    N_rounds = 1
    take_it = False

    while not(take_it):
        r_c_1,r_1_1 = [[]  for _ in range(n_agents)],[[] for _ in range(n_agents)]
        r_2_1 = [[] for _ in range(n_agents)]
        r_3_1 = [[] for _ in range(n_agents)]
        #r_4_1 = [[] for _ in range(n_agents)]
        frames = []
        imgs = []
        for i in range(max_cycles):
            actions ,_  = learner.predict(state)
            #actions ,_  = bc_policy.policy.predict(state)
            n_state,rewards ,dones ,info= env.step(actions)

            # rewards:
            with th.no_grad():
                state_t = th.Tensor(state)
                actions_t = th.Tensor(one_hot(actions,5))#np.expand_dims(actions,1))
                n_state_t = th.Tensor(n_state)
                dones_t = np.expand_dims(dones,1)

                logit_g = reward_net.group_reward(state_t.reshape(-1,*reward_net.group_obs.shape),
                            actions_t.reshape(-1,n_agents,env.action_space.n),n_state_t,dones_t)#.numpy()

                logit_s = reward_net.single_reward(state_t,actions_t,n_state_t,dones_t)#.numpy()
                x = th.hstack((logit_g.reshape(-1,1),logit_s.reshape(-1,n_agents)))
                att = reward_net.attend(x)

                for ii in range(n_agents):
                    r_c_1[ii].append(((att[:,0])*reward_net.fc.weight.T[0,ii]).numpy()) # effect of rc on r1
                    #r_1_1[ii].append(((att[:,1+ii])*reward_net.fc.weight.T[1+ii,ii]).numpy()) # effect of r1 on r1
                    r_1_1[ii].append(((att[:,1])*reward_net.fc.weight.T[1,ii]).numpy()) # effect of r1 on r1
                    r_2_1[ii].append(((att[:,2])*reward_net.fc.weight.T[2,ii]).numpy()) # effect of r1 on r1
                    r_3_1[ii].append(((att[:,3])*reward_net.fc.weight.T[3,ii]).numpy()) # effect of r1 on r1
                    #r_4_1[ii].append(((att[:,4])*reward_net.fc.weight.T[4,ii]).numpy()) # effect of r1 on r1
                #r_c_1.append((abs(att[:,0])*reward_net.fc.weight.T[0,0]).numpy()) # effect of rc on r1
                #r_1_1.append((abs(att[:,1])*reward_net.fc.weight.T[1,0]).numpy()) # effect of r1 on r1

                #breakpoint()
                x = th.mul(att,x)

                x = F.relu(reward_net.fc(x)).flatten()

                r_c = -F.logsigmoid(-logit_g) # output of group_r
                r_s = -F.logsigmoid(-logit_s) # output of single_r
                r_a = -F.logsigmoid(-att)[0]#1 batch
                r_e = -F.logsigmoid(-x)
                
                #breakpoint()

            #r_a -= r_a.min()
            r_a /= (sum(r_a))
            #breakpoint()
            #print(r_a)
            state = n_state
            env.render()

            frame = env.render(mode="rgb_array")
            imgs.append(frame.copy())
            #breakpoint()
            for m in range(0):#n_agents):
                cv2.putText(frame,str(m+1),
                    agt_pos(n_state[0,:,2:4],m), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frame = cv2.copyMakeBorder(frame,200,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))

            line = '   '.join(['--','--','r_att','r_ens'])
            line_width = 35

            cv2.putText(frame,line,
                    (0,line_width), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #breakpoint()
            line = 'r_c   {0:.2f}   {1:.2f}  -'.format(r_c[0],r_a[-1])

            cv2.putText(frame,line,
                    (0,2*line_width), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            for n in range(n_agents):
                line = 'r_s{3}   {0:.2f}   {1:.2f}   {2:.2f}'.format(r_s[n],r_a[n],r_e[n],n+1)

                cv2.putText(frame,line,
                        (0,(3+n)*line_width), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #cv2.putText(frame,'r_s {0:.2f}, r_c {1:.2f} , r_e {2:.2f}'.format(r_s[0],r_c[0],r_e[0]),
            #        (0,50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frames.append(frame)
            time.sleep(0.1)

        for jj in range(n_agents):
            vv = [np.array(r_c_1[jj]).mean(),np.array(r_1_1[jj]).mean(),np.array(r_2_1[jj]).mean(),np.array(r_3_1[jj]).mean()]
            #print(f'agent {jj} has {vv} as [coop,others...] score')
            vv1 = np.array([np.array(r_c_1[jj]),np.array(r_1_1[jj]),np.array(r_2_1[jj]),np.array(r_3_1[jj])])
            vals,counts = np.unique(vv1[:,:,0].argmin(axis=0),return_counts=True)# the least contributer
            print(f'agent {jj} has {vals[np.argmax(counts)]} as most competative [coop,others...] score')

        #breakpoint()
        r_c_ = np.array(r_c_1[0]).reshape(-1,max_cycles)
        r_1_ = np.array(r_1_1[0]).reshape(-1,max_cycles)

        if np.any((r_c_-r_1_)<1.0) and np.any((r_c_-r_1_)>7.0):
            take_it = True

    if save_video: 
        save_frames_as_gif(frames,filename='spread_gail_trained_{}_{}_{}_w.gif'.format(env_name,test_mode,
            ds_size))
    plot_rs(r_c_,r_1_,imgs,ds=ds_size,test_mode=test_mode,env_name=env_name)


if __name__=="__main__":

    main()