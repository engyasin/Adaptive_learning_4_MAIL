
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import  PPO

from multi_ppo import ppo_2_multi, unified_Mlp , MA_ActorCritic, MA_Disc,ppo_compete_ma
from make_envs import grouped_env,make_env_s,make_env_s_uni
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

n_agents = 3
max_cycles = 25
num_of_exp_eps = 400
save_video = True
SEED = 42
font = cv2.FONT_HERSHEY_SIMPLEX


def train_bc(rollouts_single):


    from imitation.algorithms import bc
    from imitation.util import logger as imit_logger

    transitions = rollout.flatten_trajectories(rollouts_single)
    venv_single_mappo = make_env_s_uni(n_agents,max_cycles)
    #format_strs = 'log'
    log_obj = imit_logger.configure(format_strs=['log'])
    bc_trainer = bc.BC(
        observation_space=venv_single_mappo.observation_space,
        action_space=venv_single_mappo.action_space,
        demonstrations=transitions,
        #policy = policy_mappo,#learner_single.policy,
        policy = MA_ActorCritic(venv_single_mappo.observation_space,
                                venv_single_mappo.action_space,
                                lr_schedule=lambda _: th.finfo(th.float32).max,
                                n_agents=n_agents),
        device = "cuda",
        custom_logger = log_obj,
        #rng=rng,
    )

    #bc_trainer.train(n_epochs=1)#20

    return bc_trainer

global_expert = expert_2_multi(n_agents=n_agents)
rollouts_coop = rollout.rollout(
    global_expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(grouped_env(SEED,n_agents,max_cycles))] * 8),
    rollout.make_sample_until(min_timesteps=None, min_episodes=num_of_exp_eps),
)

rollouts_single = from_many_to_single(rollouts_coop,n_agents=n_agents,ma=True)

bc_policy = train_bc(rollouts_single)

class policy_mappo(MA_ActorCritic):

    def __init__(self, observation_space, action_space, lr_schedule, n_agents=n_agents, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, n_agents,
            net_arch = [dict(pi=[128, 64], vf=[64, 64])],**kwargs)

venv_single_mappo = make_env_s_uni(n_agents,max_cycles)
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

learner_single = PPO.load('learner_trained_{0}_with_{1}_data.pth'.format(n_agents,
    num_of_exp_eps))

reward_net_single = MA_Disc(
    venv_single_mappo.observation_space, venv_single_mappo.action_space, normalize_input_layer=RunningNorm
)

reward_net_single = (th.load('spread_{0}_agents_{1}_exps_rew_net.pth'.format(
            n_agents,num_of_exp_eps
        ),map_location=th.device('cpu'))).eval()

venv_coop_m = DummyVecEnv([lambda :grouped_env(SEED,n_agents,max_cycles)] * n_agents)

reward_net_coop = BasicRewardNet(
    venv_coop_m.observation_space, venv_coop_m.action_space, normalize_input_layer=RunningNorm
)

reward_net_coop = (th.load('spread_{0}_agents_{1}_exps_rew_net_coop.pth'.format(
            n_agents,num_of_exp_eps
        ),map_location=th.device('cpu'))).eval()





EnsRew = DsEnsemble(n_agents=n_agents,lr=0.05)#.cuda()
EnsRew.load_state_dict(th.load('spread_{0}_agents_{1}_exps_ensnet.pth'.format(
            n_agents,num_of_exp_eps
        ),map_location=th.device('cpu')))

eval_env = make_env_s_uni(n_agents,max_cycles)

learner_c_rewards, learner_std = evaluate_policy(
    learner_single, eval_env, 100,# return_episode_rewards=True
)

print('expert: ',-20)


print(learner_c_rewards) # -33,5
print(learner_std)



r_single =lambda *kwargs: -F.logsigmoid(-1*reward_net_single(*kwargs))
r_coop_raw = lambda *kwargs: -F.logsigmoid(-1*reward_net_coop(*kwargs))

r_coop = fix_coop(r_coop_raw,
    n_agents= n_agents,ma=True)

ens_rew = ens_rews_full(r_single,
    r_coop_raw,
    EnsRew.eval(),n_agents= n_agents,
    ma=True)

ens_rew_att = ens_rews_full(r_single,
    r_coop_raw,
    EnsRew.attend.eval(),n_agents= n_agents,
    ma=True)

bc_epochs = 0
for i in range(bc_epochs):
    bc_c_rewards, bc_std = evaluate_policy(
        bc_policy.policy, eval_env, 100,# return_episode_rewards=True
    )
    print("bc epoch {}".format(i+1))
    print(bc_c_rewards) # mean (-30-32) (5 std)
    print(bc_std)
    bc_policy.train(n_epochs=1)#20



frames = []
state = venv_single_mappo.reset()
for i in range(max_cycles*15):
    actions ,_  = learner_single.predict(state)
    #actions ,_  = bc_policy.policy.predict(state)
    n_state,rewards ,dones ,info= venv_single_mappo.step(actions)

    # rewards:
    with th.no_grad():
        state_t = th.Tensor(state)
        actions_t = th.Tensor(one_hot(actions,5))#np.expand_dims(actions,1))
        n_state_t = th.Tensor(n_state)
        dones_t = np.expand_dims(dones,1)

        r_c = r_coop(state_t,actions_t,n_state_t,dones_t)#.numpy()
        r_s = r_single(state_t,actions_t,n_state_t,dones_t).numpy()

        r_e = ens_rew(state_t,actions_t,n_state_t,dones_t)#.numpy()

        r_a = ens_rew_att(state_t,actions_t,n_state_t,dones_t)#.numpy()

    #r_a -= r_a.min()
    r_a /= (sum(r_a))
    #breakpoint()
    #print(r_a)
    state = n_state
    venv_single_mappo.render()

    frame = venv_single_mappo.render(mode="rgb_array")
    #breakpoint()
    for m in range(0):#n_agents):
        cv2.putText(frame,str(m+1),
            agt_pos(n_state[0,:,2:4],m), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    frame = cv2.copyMakeBorder(frame,200,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))

    line = '   '.join(['--','--','r_att','r_ens'])
    line_width = 35

    cv2.putText(frame,line,
            (0,line_width), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
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
    #time.sleep(0.1)

if save_video: 
    save_frames_as_gif(frames,filename='spread_gail_trained_with_rs_{}_{}.gif'.format(n_agents,
        num_of_exp_eps))



