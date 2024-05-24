
import gym

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class unified_env_soccer(gym.Env):
    """
    Multi-agent (cooperative) env from single env
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, env):
        super(unified_env_soccer, self).__init__()
        self.env = env
        #self.env.__init__()

        self.action_space = gym.spaces.MultiDiscrete([self.env.action_space.n]*self.env.num_envs)
        #continues
        self.observation_space = gym.spaces.Box(0,128,
                                                (np.prod(self.env.observation_space.shape)*self.env.num_envs,),
                                                np.uint8)
        self.num_envs = 1
        """


        old_obs = self.env.observation_space
        self.old_shape = old_obs.shape
        last_dim = old_obs.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=old_obs.low.min(),
            high=old_obs.high.max(),
            shape=(*old_obs.shape[:-1],last_dim*self.env.num_envs), # 5,5,6*4
            dtype=old_obs.dtype)
        self.num_envs = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        """


    def step(self, action):
        # Execute one time step within the environment
        # already vectorized (n_agents,1)
        next_state, reward, done, infos = self.env.step(action)

        n_info = infos[0]
        if 'terminal_observation' in n_info:
            n_info['terminal_observation'] = n_info['terminal_observation'].flatten()

        for info in infos[1:]:
            for k,v in info.items():
                n_info.update({k:np.hstack((n_info[k],v.flatten()))})
        #n_obs = next_state.reshape((*self.old_shape[:-1],self.old_shape[-1]*self.env.num_envs))
        #n_obs = np.array([np.dstack([next_state[i+j] for i in range(self.env.num_envs)]) 
        #            for j in range(next_state.shape[0]//self.env.num_envs)])
                  #for nobs in next_state[::self.env.num_envs]])

        #breakpoint()#n_agents*obs_shape
        return  next_state.flatten(), reward.sum(), done.all(), n_info

    def reset(self):
        # Reset the state of the environment to an initial state
        new_obs = self.env.reset()

        #obs = np.array([np.dstack([new_obs[i+j] for i in range(self.env.num_envs)]) 
        #            for j in range(new_obs.shape[0]//self.env.num_envs)])
        #print(new_obs.shape)
        #obs = new_obs.reshape(self.old_shape[0])
        return new_obs.flatten()
        #new_obs.reshape((*self.old_shape[:-1],self.old_shape[-1]*self.env.num_envs))#flatten()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.env.render( mode=mode, close=close)
class unified_env(gym.Env):
    """
    Multi-agent (cooperative) env from single env
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, env):
        super(unified_env, self).__init__()
        self.env = env
        #self.env.__init__()

        self.action_space = gym.spaces.MultiDiscrete([self.env.action_space.n]*self.env.num_envs)
        #continues
        obs_l = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,
                                                (np.prod(self.env.observation_space.shape)*self.env.num_envs,),np.float32)
        self.num_envs = 1
        """


        old_obs = self.env.observation_space
        self.old_shape = old_obs.shape
        last_dim = old_obs.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=old_obs.low.min(),
            high=old_obs.high.max(),
            shape=(*old_obs.shape[:-1],last_dim*self.env.num_envs), # 5,5,6*4
            dtype=old_obs.dtype)
        self.num_envs = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        """


    def step(self, action):
        # Execute one time step within the environment
        # already vectorized (n_agents,1)
        next_state, reward, done, infos = self.env.step(action)

        n_info = infos[0]
        if 'terminal_observation' in n_info:
            n_info['terminal_observation'] = n_info['terminal_observation'].flatten()

        for info in infos[1:]:
            for k,v in info.items():
                n_info.update({k:np.hstack((n_info[k],v.flatten()))})
        #n_obs = next_state.reshape((*self.old_shape[:-1],self.old_shape[-1]*self.env.num_envs))
        #n_obs = np.array([np.dstack([next_state[i+j] for i in range(self.env.num_envs)]) 
        #            for j in range(next_state.shape[0]//self.env.num_envs)])
                  #for nobs in next_state[::self.env.num_envs]])

        #breakpoint()#n_agents*obs_shape
        return  next_state.flatten(), reward.sum(), done.all(), n_info

    def reset(self):
        # Reset the state of the environment to an initial state
        new_obs = self.env.reset()

        #obs = np.array([np.dstack([new_obs[i+j] for i in range(self.env.num_envs)]) 
        #            for j in range(new_obs.shape[0]//self.env.num_envs)])
        #print(new_obs.shape)
        #obs = new_obs.reshape(self.old_shape[0])
        #breakpoint()
        return new_obs.flatten()
        #new_obs.reshape((*self.old_shape[:-1],self.old_shape[-1]*self.env.num_envs))#flatten()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.env.render( mode=mode, close=close)





class unified_obs(gym.Env):

    """
    Multi-agent env from single env
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, env):
        super(unified_obs, self).__init__()
        self.env = env
        #self.env.__init__()

        self.action_space = self.env.action_space
        #gym.spaces.MultiDiscrete([self.env.action_space.n]*self.env.num_envs)
        #continues
        obs_l = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(obs_l,self.env.num_envs),np.float32)
        self.num_envs = self.env.num_envs
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

    def step(self, action):
        # Execute one time step within the environment

        assert (action.shape[0]%self.num_envs)==0

        next_states, reward, done, infos = self.env.step(action)

        m_states = next_states.reshape(next_states.shape[0]//self.num_envs
        ,self.num_envs,-1)

        processed_states = self.process_states(m_states)

        return processed_states, reward, done, infos


    def reset(self):
        # Reset the state of the environment to an initial state
        new_obs = self.env.reset()

        m_states = new_obs.reshape(new_obs.shape[0]//self.num_envs
        ,self.num_envs,-1)

        processed_states = self.process_states(m_states)
        #print(new_obs.shape)
        return processed_states

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.env.render( mode=mode, close=close)


    def process_states(self,states):

        #NOTE arrangemnt is not with minemal change in position
        result = []
        for one_step in states:
            for i in range(one_step.shape[0]):
                result.append(np.roll(one_step,-1*i,axis=0))

        return np.array(result)



class VecExtractDictObs(VecEnvWrapper):
    """
    convert single obs into global envs for MAPPO value training
    number of envs should equal number of agents

    Add eps_reward and eps_len
    """

    def __init__(self, venv: VecEnv):

        obs_l = venv.observation_space.shape[0]#dim of vector obs
        super().__init__(venv=venv, observation_space= gym.spaces.Box(-np.inf,np.inf,(venv.num_envs,obs_l),np.float32),
            action_space = venv.action_space)

        self.ep_r = np.zeros(venv.num_envs)
        self.ep_l = 0
        self.t = 0

    def reset(self) -> np.ndarray:


        # Reset the state of the environment to an initial state
        new_obs = self.venv.reset()

        m_states = new_obs.reshape(new_obs.shape[0]//self.venv.num_envs
        ,self.venv.num_envs,-1)

        processed_states = self.process_states(m_states)
        #print(new_obs.shape)

        return processed_states

    def step_backup(self, action):
        # Execute one time step within the environment

        next_states, reward, done, infos = self.venv.step(action)

        m_states = next_states.reshape(next_states.shape[0]//self.num_envs
        ,self.num_envs,-1)

        processed_states = self.process_states(m_states)

        return processed_states, reward, done, infos

    def step_async(self, actions: np.ndarray) -> None:

        #assert (actions.shape[0]%self.num_envs)==0
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:

        obs, reward, done, info = self.venv.step_wait()
        self.t += 1
        #breakpoint()
        # return to env_utils in il_test line 174
        if done.all():
            #breakpoint()
            first_env = np.array([[inf["terminal_observation"] for inf in info]])
            all_envs = self.process_states(first_env)
            for i,env_terminal in enumerate(all_envs):
                info[i]['terminal_observation'] = env_terminal

            for i,f in enumerate(info):
                f.update({'episode': {'r':self.ep_r[i],
                         'l': self.ep_l, 't': self.t}})
            self.ep_l = 0
            self.ep_r = np.zeros_like(reward)

        elif done.any():
            raise 'Some env are done but not all'
        else:
            self.ep_l += 1
            self.ep_r += reward

        m_states = obs.reshape(obs.shape[0]//self.venv.num_envs
            ,self.venv.num_envs,-1)

        processed_states = self.process_states(m_states)

        return processed_states, reward, np.array(done,dtype=bool), info

    def process_states(self,states):

        #NOTE arrangemnt is not with minemal change in position
        # state: batch, num_env, obs_len
        result = []
        for one_step in states:
            for i in range(self.venv.num_envs):
                result.append(np.roll(one_step,-1*i,axis=0))

        return np.array(result)


class Vecbooldone_Soccer(VecEnvWrapper):
    """
    make done bool
    """

    def __init__(self, n_agents, venv: VecEnv,flatt_obs=True,**kwargs):

        super().__init__(venv=venv,**kwargs )

        self.ep_r = np.zeros(venv.num_envs)
        self.ep_l = 0
        self.t = 0
        self.flatt_obs = flatt_obs
        if flatt_obs:
            #5,5,6
            old_obs = self.observation_space
            #TODO last_dim -1
            last_dim = old_obs.shape[-1]
            self.observation_space = gym.spaces.Box(
                low=old_obs.low.min(),
                high=old_obs.high.max(),
                shape=(np.prod(old_obs.shape[:-1])*(last_dim),), # 5,5,6
                dtype=old_obs.dtype)
            #NOTE for soccer
        self.n_agents = n_agents

    def flatten(self,arr):
        if self.flatt_obs:
            return arr.reshape(arr.shape[0],-1).copy()
        else:
            return arr.copy()

    def reset(self,seed=42) -> np.ndarray:

        #self.prev_obs = self.flatten(self.venv.reset())

        # Reset the state of the environment to an initial state
        obs_list = []
        try:
            for i,venv_ in enumerate(self.venv.unwrapped.vec_envs):
                obs = venv_.par_env.reset(seed=seed+i)
                for k,v in obs.items():
                    obs_list.append(v.copy())
                #venv_.par_env.unwrapped.original_env.seed(seed=seed+i)
                #obs = venv_.par_env.unwrapped.original_env.reset()
                #obs_list.extend(obs)
            self.prev_obs = self.flatten(np.array(obs_list,dtype=np.uint8))
            return self.flatten(np.array(obs_list,dtype=np.uint8))
        except:
            self.prev_obs = self.flatten(self.venv.reset())

        return  self.prev_obs

    def step_async(self, actions: np.ndarray) -> None:

        #assert (actions.shape[0]%self.num_envs)==0
        self.c_acts = np.array(actions)
        self.venv.step_async(actions)
        

    def step_wait(self) -> VecEnvStepReturn:

        obs, reward, done, info = self.venv.step_wait()
        obs = self.flatten(obs)
        self.t += 1
        #breakpoint()
        # return to env_utils in il_test line 174
        if done.all():
            #breakpoint()
            for i,f in enumerate(info):
                f.update({'episode': {'r':self.ep_r[i],
                         'l': self.ep_l, 't': self.t}})
            self.ep_l = 0
            self.ep_r = np.zeros_like(reward)

        elif done.any():
            #breakpoint()
            raise 'Some env are done but not all'
        else:
            self.ep_l += 1
            self.ep_r += reward

        n_obs = obs.reshape(obs.shape[0]//self.n_agents,self.n_agents,*obs.shape[1:])
        p_obs = self.prev_obs.reshape(obs.shape[0]//self.n_agents,self.n_agents,*obs.shape[1:]).copy()
        self.prev_obs = obs.copy()
        # true_batch_size,n_agents,obs_shape
        n_acts = self.c_acts.reshape(self.c_acts.shape[0]//self.n_agents,self.n_agents,-1)
        n_info = []

        last_obs = []
        for i,f in enumerate(info):
            if (f.keys()):
                f['terminal_observation'] = f['terminal_observation'].flatten()
                last_obs.append(f['terminal_observation'])
            n_info.append({
            'group_obs':p_obs[i//self.n_agents,...].copy(),#4,150
            'group_acts':n_acts[i//self.n_agents,...].copy(),
            'group_next_obs':n_obs[(i//self.n_agents),...].copy(),
            **f,
            })
        if last_obs:
            f_obs = []
            for i in range(obs.shape[0]//self.n_agents):
                f_obs.append(np.vstack(last_obs[self.n_agents*(i):self.n_agents*(i+1)]))
            for j,f in enumerate(n_info):
                f['group_next_obs'] = f_obs[j//self.n_agents].copy()
        #if done.all():
        #    breakpoint()
        return  obs, reward, np.array(done).astype(bool), n_info
    def reset_set_state(self,states):
        _ = self.reset()

        fix_landmarks = self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.landmarks[:]
        fix_landmarks.extend([fix_landmarks[-1]])

        for i,state in enumerate(states):
            self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.agents[i].state.p_pos = state[0]
            self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.agents[i].state.p_vel = state[1]
            fix_landmarks[i].state.p_pos = state[2]

    def get_state(self):
        fix_landmarks = self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.landmarks[:]
        fix_landmarks.extend([fix_landmarks[-1]])
        states = []
        for i,agent in enumerate(self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.agents):
            states.append(np.vstack([agent.state.p_pos,agent.state.p_vel,
                    fix_landmarks[i].state.p_pos]).copy())
        return np.array(states).copy()

    def get_positions(self):
        positions = []
        for agent in self.venv.unwrapped.vec_envs[0].par_env.unwrapped.original_env.agents:
            positions.append(agent.pos.copy())

        #positions.append(self.venv.unwrapped.vec_envs[0].par_env.unwrapped.original_env.ball.pos.copy())
        return np.array(positions)

    def render(self,mode='human'):

        return self.venv.render(mode=mode)

class Vecbooldone(VecEnvWrapper):
    """
    make done bool
    """

    def __init__(self, n_agents, venv: VecEnv,**kwargs):

        super().__init__(venv=venv,**kwargs )

        self.ep_r = np.zeros(venv.num_envs)
        self.ep_l = 0
        self.t = 0

        self.n_agents = n_agents

    def reset(self,seed=0) -> np.ndarray:

        # Reset the state of the environment to an initial state
        self.prev_obs = self.venv.reset()
        return  self.prev_obs

    def step_async(self, actions: np.ndarray) -> None:

        #assert (actions.shape[0]%self.num_envs)==0
        self.c_acts = np.array(actions)
        self.venv.step_async(actions)
        

    def step_wait(self) -> VecEnvStepReturn:

        obs, reward, done, info = self.venv.step_wait()
        self.t += 1
        #breakpoint()
        # return to env_utils in il_test line 174
        if done.all():
            #breakpoint()
            for i,f in enumerate(info):
                f.update({'episode': {'r':self.ep_r[i],
                         'l': self.ep_l, 't': self.t}})
            self.ep_l = 0
            self.ep_r = np.zeros_like(reward)

        elif done.any():
            raise 'Some env are done but not all'
        else:
            self.ep_l += 1
            self.ep_r += reward


        n_obs = obs.reshape(obs.shape[0]//self.n_agents,self.n_agents,*obs.shape[1:])
        p_obs = self.prev_obs.reshape(obs.shape[0]//self.n_agents,self.n_agents,*obs.shape[1:]).copy()
        self.prev_obs = obs.copy()
        # true_batch_size,n_agents,obs_shape
        n_acts = self.c_acts.reshape(self.c_acts.shape[0]//self.n_agents,self.n_agents,-1)
        n_info = []
        for i,f in enumerate(info):
            #if (f.keys()):
            #    breakpoint()
            #if len(f.keys()):#(i//self.n_agents)==(n_obs.shape[0]-1) or 
            #    n_info.append({**f})
            #else:
            n_info.append({
            'group_obs':p_obs[i//self.n_agents,...].copy(),
            'group_acts':n_acts[i//self.n_agents,...].copy(),
            'group_next_obs':n_obs[(i//self.n_agents),...].copy(),
            **f,
            })
        #breakpoint()
        return  obs, reward, np.array(done,dtype=bool), n_info

    def reset_set_state(self,states):
        _ = self.reset()

        fix_landmarks = self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.landmarks[:]
        fix_landmarks.extend([fix_landmarks[-1]])

        for i,state in enumerate(states):
            self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.agents[i].state.p_pos = state[0]
            self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.agents[i].state.p_vel = state[1]
            fix_landmarks[i].state.p_pos = state[2]

    def get_state(self):
        fix_landmarks = self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.landmarks[:]
        fix_landmarks.extend([fix_landmarks[-1]])
        states = []
        for i,agent in enumerate(self.venv.unwrapped.vec_envs[0].par_env.aec_env.unwrapped.world.agents):
            states.append(np.vstack([agent.state.p_pos,agent.state.p_vel,
                    fix_landmarks[i].state.p_pos]).copy())
        return np.array(states).copy()