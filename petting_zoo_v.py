import functools

import gym
from gym.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers



def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_soccer(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_soccer(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, env_constructor, n_agents=4, max_cycles=55, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(n_agents)]
        #self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

        self.max_itr = max_cycles

        self.original_env = env_constructor()

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.original_env.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.original_env.action_space

    def render(self,mode='human'):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        self.render_mode = mode
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        img = self.original_env.render(mode)
        if mode == 'rgb_array':
            return img

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.original_env.seed(seed=seed)
        obs_list = self.original_env.reset()
        obs_list = self.homogenous_obs(obs_list)
        observations = {agent: obs_list[i] for i,agent in enumerate(self.agents)}
        return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        
        obs, reward, done, info = self.original_env.step(list(actions.values()))
        obs = self.homogenous_obs(obs)
        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: reward[i] for i,agent in enumerate(self.agents)}

        self.num_moves += 1

        terminations = {agent: done or(self.num_moves >= self.max_itr) for agent in self.agents}


        #truncations = {agent: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]:obs[i] 
            for i in range(len(self.agents))
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if any(list(terminations.values())):
            # new cycle of the game
            self.agents = []
            for k,v in infos.items():
                v.update({'terminal_observation':observations[k].copy()})
            
            observations = self.reset()

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations,  infos
    
    def homogenous_obs(self,obs_list):
        # make oneself team 1 and opponent team 2 always
        new_obs_list = []
        for obs_ in obs_list:
            obs = obs_.copy().T
            if obs[1,-1,obs.shape[2]//2] != 1: #or == 2
                obs[1][obs_.T[1] == 2] = 1
                obs[1][obs_.T[1] == 1] = 2
            new_obs_list.append(obs.T)
        return new_obs_list
        

