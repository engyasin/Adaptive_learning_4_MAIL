
from dis import dis
import numpy as np
from typing import Callable

from stable_baselines3.common.policies import BasePolicy

# 4 agents
#ob space [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
#actions [no_action, move_left, move_right, move_down, move_up]


def eclud_d(a,b):
    x = abs(a[0]-b[0])
    y = abs(a[1]-b[1])

    return np.linalg.norm((x,y))

class expert_p(Callable):

    def __init__(self, n_agents = 4) -> None:

        self.n_agents = n_agents

        self.dists_mat = np.zeros((n_agents,n_agents))

        self.pnts_acts = np.zeros((2,n_agents),dtype=np.int64)-1

        self.agents = []
        self.pnts = []

        self.acts = []

    def __call__(self, obs,**kwargs):
        return self.predict(obs,**kwargs)[0]

    def predict(self,obs,**kwargs):

        acts = []
        for obs_single in obs:
            acts.append(self.predict_single(obs_single))

        return acts, None

    def predict_single(self,obs_s):
        #ob space [self_vel, self_pos, 
        # landmark_rel_positions, other_agent_rel_positions, 
        # communication]

        self.agents = np.hstack(([0,0],
            obs_s[(self.n_agents-1)*-4:(self.n_agents-1)*-2])).reshape(-1,2)
        self.pnts = obs_s[4:(self.n_agents*2)+4].reshape(-1,2)

        # calculate dist_matrix
        D = self.find_dists_mat(self.agents,self.pnts)

        # find nearst pnt
        n_pnt = self.nearst_pnt(D)

        # best actions
        self.acts = []
        for i,agnt in enumerate(self.pnts_acts[0,:]):
            self.acts.append(self.best_action(i,self.agents[int(agnt)]))


        # step
        new_agents = self.step()

        # check collide
        if self.check_collid(new_agents):
            n_pnt = self.pnts_acts[0,:].argmin()
            x,y = self.pnts[n_pnt] 
            new_act = [(x>0)+1,(y>0)+3][self.acts[n_pnt]<3]
            self.acts[n_pnt] = new_act

            # once more
            new_agents = self.step()
            if self.check_collid(new_agents):
                # no action
                self.acts[n_pnt] = 0

        n_pnt = self.pnts_acts[0,:].argmin()
        return self.acts[n_pnt]

    def find_dists_mat(self,agents,pnts):

        for i,agent in enumerate(agents):
            for j,pnt in enumerate(pnts):
                self.dists_mat[i,j] = eclud_d(agent,pnt)

        return self.dists_mat.copy()


    def nearst_pnt(self,D):

        indx = np.unravel_index(D.argmin(),D.shape)

        agnts_idx = [i for i in range(self.n_agents)]
        _ = agnts_idx.pop(agnts_idx.index(indx[0]))
        while len(agnts_idx):
            #breakpoint()
            self.pnts_acts[0,indx[1]] = indx[0]

            # delete agent
            D[indx[0],:] = 1e5
            D[:,indx[1]] = 1e5

            # more than one min?:
            indx = np.unravel_index(D.argmin(),D.shape)

            D_mins = (D==D.min())
            if D_mins.sum()>1:
                rs,cs = np.where(D_mins) 
                nxt_min = np.array([D[r,:][np.logical_not(D_mins[r,:])].min()
                 for r in rs])
                
                indx = rs[nxt_min.argmax()],cs[nxt_min.argmax()]
                    
            _ = agnts_idx.pop(agnts_idx.index(indx[0]))

        #breakpoint()
        # last agent
        self.pnts_acts[0,indx[1]] = indx[0]
        return self.pnts_acts[0,:].argmin()


    def best_action(self,n_pnt,agent):
        #actions [no_action, move_left, move_right, move_down, move_up]

        x,y = self.pnts[n_pnt] - agent
        if (abs(x)+abs(y))<0.01:
            act = 0
        else:
            act = [(x>0)+1,(y>0)+3][abs(x)<abs(y)]

        return act

    def check_collid(self,new_agents,radius=0.2):

        for i in range(1,self.n_agents):
            if eclud_d(new_agents[0],new_agents[i])<(radius*2):
                return True
            
        return False

    def step(self):
        vx,vy = 1,1
        new_agents = self.agents.copy()

        for i,act in enumerate(self.acts):
            if act == 1:
                new_agents[i] -= np.array([vx,0])
            elif act == 2:
                new_agents[i] += np.array([vx,0])
            elif act == 3:
                new_agents[i] -= np.array([0,vy])
            elif act == 4:
                new_agents[i] += np.array([0,vy])
            else:
                pass

        return new_agents

class expert_adv(expert_p):

    def __init__(self, n_agents = 4, **kwargs) -> None:
        super().__init__(n_agents=n_agents-1,**kwargs)

    def predict_single(self,obs_s):
        if any(obs_s[-2:]):
            #n_obs_s1 = np.insert(obs_s[2:],2,obs_s[:2])
            # to imitate communications
            n_obs_s = np.hstack((obs_s,np.zeros(self.n_agents*2-2)))
            return super().predict_single(n_obs_s)
        else:
            # adv obs = [landmark_rel_position, other_agents_rel_positions,0,0]
            marks,agents = tuple(obs_s[:-2].reshape((2,-1)))
            m_agnt = agents.reshape((-1,2)).mean(axis=0)
            idx = np.argmin([eclud_d(pnt,m_agnt) for pnt in marks.reshape(-1,2)])
            x,y = marks.reshape(-1,2)[idx]
            act = [(x>0)+1,(y>0)+3][abs(x)<abs(y)] 
            return act

    def find_dists_mat(self, agents, pnts):
        dists_mat = super().find_dists_mat(agents, pnts)
        #breakpoint()
        dists_mat[dists_mat != dists_mat[:,0].min()] += 10
        return dists_mat

class expert_multi(Callable):
    def __init__(self, expert, n_agents = 4, **kwargs):
        #super().__init__(*args, squash_output=squash_output, **kwargs)
        self.n_agents = n_agents
        self.expert = expert

    def predict(self,states, **kwargs):
        #print(states.shape)
        #batch size 1

        n_states = states.reshape(-1,#self.n_agents,
                                  *self.expert.observation_space.shape)
        acts, _ = self.expert.predict(n_states,deterministic=True)
        acts = np.array(acts).reshape(-1,self.n_agents)#,
        #*self.expert.action_space.shape)

        return acts, None

    def __call__(self, obs,**kwargs):
        return self.predict(obs,**kwargs)[0]


class expert_2_multi(expert_p):
    def __init__(self, n_agents = 4, **kwargs) -> None:
        super().__init__(n_agents=n_agents,**kwargs)


    def predict(self,states, **kwargs):
        #print(states.shape)

        # from group to single
        n_states = states.reshape(states.shape[0]*self.n_agents,-1)

        acts, _ = super().predict(n_states,**kwargs)

        acts = np.array(acts).reshape(states.shape[0],self.n_agents)


        return acts, None

    def __call__(self, obs,**kwargs):
        return self.predict(obs,**kwargs)[0]


class expert_2_multi_adv(expert_adv):
    def __init__(self, n_agents = 4, **kwargs) -> None:
        super().__init__(n_agents=n_agents,**kwargs)


    def predict(self,states, **kwargs):
        #print(states.shape)
        n_states = states.reshape(states.shape[0]*(self.n_agents+1),-1)

        acts, _ = super().predict(n_states,**kwargs)

        acts = np.array(acts).reshape(states.shape[0],(self.n_agents+1))

        return acts, None

    def __call__(self, obs,**kwargs):
        return self.predict(obs,**kwargs)[0]

    """
    def set_env(self, env, **kwargs) -> None:
        try:
            super().set_env(env, **kwargs)
        except ValueError:
            print('Env error passed!')
            self.new_env = env
            self._obs = self.new_env.reset()
            #self.env.num_envs = self.n_envs = env.num_envs
        return None

    def simulte_trajs(self,trainer,total_timesteps=10):

        for i in range(total_timesteps):
            action, _state = self.predict(self._obs, deterministic=True)
            self._obs, reward, done, info = self.new_env.step(action)

            #if done.all():
            #    self._obs = self.new_env.reset()

        gen_trajs, ep_lens = trainer.venv_buffering.pop_trajectories()
        trainer._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        trainer._gen_replay_buffer.store(gen_samples)

        return trainer
    """

