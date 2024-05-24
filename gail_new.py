from imitation.algorithms.adversarial.gail import GAIL
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type

import torch as th
from stable_baselines3.common import base_class, vec_env
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets
from imitation.data import types
from imitation.util import logger, networks, util

import tqdm
import numpy as np
import random
from imitation.algorithms.adversarial.airl import AIRL


class AIRL_ENS(AIRL):
    
    def __init__(self, *, demonstrations: base.AnyTransitions, demo_batch_size: int, venv: vec_env.VecEnv, gen_algo: base_class.BaseAlgorithm, reward_net: reward_nets.RewardNet,
                 n_agents=4 ,**kwargs):
        
        super().__init__(demonstrations=demonstrations,
             demo_batch_size=demo_batch_size,
              venv=venv, gen_algo=gen_algo, reward_net=reward_net, **kwargs)

        self.old_demo_batch_size = demo_batch_size
        self.n_agents = n_agents

    def flatten_samples(self,expert_samples,gen_samples):

        expert_samples['obs'] = np.vstack([f['group_obs'] for f in expert_samples['infos']])
        expert_samples['acts'] = np.vstack([f['group_acts'] for f in expert_samples['infos']])#.T[0]
        expert_samples['next_obs'] = np.vstack([f['group_next_obs'] for f in expert_samples['infos']])
        expert_samples['dones'] = np.array([[f['done'] for _ in range(self.n_agents)] for f in expert_samples['infos']],dtype=bool)

    def fix_shape(self,arr):
        #just for soccer
        res = []
        l_dim = arr.shape[-1]//self.n_agents
        for i in range(0,arr.shape[-1],l_dim):
            res.append(arr[...,i:i+l_dim])
        return np.vstack(res)

    def prepare_for_disc(
        self,
    ):

        expert_samples = self._next_expert_batch()

        if self._gen_replay_buffer.size() == 0:
            raise RuntimeError(
                "No generator samples for training. " "Call `train_gen()` first.",
            )
        
        gen_samples = self._gen_replay_buffer.sample(self.old_demo_batch_size*2)
        gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_agents = self.n_agents

        
        #random.shuffle(expert_samples['infos'])
        #breakpoint()
        #expert_samples['obs'] = np.vstack([self.fix_shape(f['group_obs']) for f in expert_samples['infos']])
        #expert_samples['acts'] = np.vstack([self.fix_shape(f['group_acts']) for f in expert_samples['infos']]).T[0]
        #expert_samples['next_obs'] = np.vstack([self.fix_shape(f['group_next_obs']) for f in expert_samples['infos']])
        expert_samples['dones'] = np.array([[f['done'] for _ in range(n_agents)] for f in expert_samples['infos']],dtype=bool)
        #expert_samples['next_obs'] = expert_samples['next_obs'].detach().numpy().repeat(n_agents,axis=0)
        expert_samples['obs'] = np.vstack([f['group_obs'].reshape(n_agents,-1) for f in expert_samples['infos']])
        expert_samples['acts'] = np.hstack([f['group_acts'] for f in expert_samples['infos']])#.T[0]
        expert_samples['next_obs'] = np.vstack([f['group_next_obs'].reshape(n_agents,-1) for f in expert_samples['infos']])


        gen_samples['infos'] = gen_samples['infos'].tolist()
        #breakpoint()
        #random.shuffle(gen_samples['infos'])original_env_rew
        #new_r = sorted(np.array([f['new_rew'] for f in gen_samples['infos']]),reverse=True)
        #old_r = sorted(np.array([f['original_env_rew'] for f in gen_samples['infos']]),reverse=True)

        #rr = sum(new_r[:self.old_demo_batch_size])/sum(new_r[self.old_demo_batch_size:])
        #rrr = sum(old_r[:self.old_demo_batch_size])/sum(old_r[self.old_demo_batch_size:])

        #gen_samples['infos'] = sorted(gen_samples['infos'],key=lambda x: x['new_rew'], reverse=True)[:self.old_demo_batch_size]
        
        gen_obs = []
        gen_next_obs = []
        gen_acts = []
        dones = []
        for f in gen_samples['infos']:
            if 'group_obs' in f:
                gen_obs.append(f['group_obs']) 
                gen_next_obs.append(f['group_next_obs'])
                gen_acts.append(f['group_acts'])
                dones.append([('terminal_observation' in f) for _ in range(n_agents)])
            if len(gen_obs)== self.old_demo_batch_size:
                break

        #breakpoint()
        gen_samples['obs'] = np.vstack(gen_obs)
        gen_samples['acts'] = np.vstack(gen_acts).T[0]
        gen_samples['next_obs'] = np.vstack(gen_next_obs)
        gen_samples['dones'] = np.array(dones,dtype=bool)

        #gen_samples['next_obs'] = gen_samples['next_obs'][:self.old_demo_batch_size].repeat(n_agents,axis=0)

        self.demo_batch_size = self.old_demo_batch_size * n_agents
        #TODO change obs, acts to be grouped togther
        #breakpoint()
        return {'expert_samples':expert_samples,'gen_samples':gen_samples},f'{0} ** {0}'


    @property
    def new_reward(self):# -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        #group_reward_net = self._reward_net.group_reward.base
        #single_reward_net = self._reward_net.single_reward.base
        #def new_blended_reward(state, action, next_state, done):
        #    pass
        # Recursively return the base network of the wrapped reward net
        #while isinstance(reward_net, reward_nets.RewardNetWrapper):
        #    reward_net = reward_net.base
        return self._reward_net.new_reward


class GAIL_ENS(GAIL):
    
    def __init__(self, *, demonstrations: base.AnyTransitions, demo_batch_size: int, venv: vec_env.VecEnv, gen_algo: base_class.BaseAlgorithm, reward_net: reward_nets.RewardNet, **kwargs):
        
        super().__init__(demonstrations=demonstrations,
             demo_batch_size=demo_batch_size,
              venv=venv, gen_algo=gen_algo, reward_net=reward_net, **kwargs)

        self.old_demo_batch_size = demo_batch_size


    def prepare_for_disc(
        self,
    ):

        expert_samples = self._next_expert_batch()

        if self._gen_replay_buffer.size() == 0:
            raise RuntimeError(
                "No generator samples for training. " "Call `train_gen()` first.",
            )
        
        gen_samples = self._gen_replay_buffer.sample(self.old_demo_batch_size*2)
        gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_agents = expert_samples['infos'][0]['group_obs'].shape[0]

        
        #random.shuffle(expert_samples['infos'])
        expert_samples['obs'] = np.vstack([f['group_obs'] for f in expert_samples['infos']])
        expert_samples['acts'] = np.vstack([f['group_acts'] for f in expert_samples['infos']]).T[0]
        expert_samples['next_obs'] = np.vstack([f['group_next_obs'] for f in expert_samples['infos']])


        gen_samples['infos'] = gen_samples['infos'].tolist()
        #breakpoint()
        #random.shuffle(gen_samples['infos'])original_env_rew
        #new_r = sorted(np.array([f['new_rew'] for f in gen_samples['infos']]),reverse=True)
        #old_r = sorted(np.array([f['original_env_rew'] for f in gen_samples['infos']]),reverse=True)

        #rr = sum(new_r[:self.old_demo_batch_size])/sum(new_r[self.old_demo_batch_size:])
        #rrr = sum(old_r[:self.old_demo_batch_size])/sum(old_r[self.old_demo_batch_size:])

        #gen_samples['infos'] = sorted(gen_samples['infos'],key=lambda x: x['new_rew'], reverse=True)[:self.old_demo_batch_size]
        
        gen_obs = []
        gen_next_obs = []
        gen_acts = []
        for f in gen_samples['infos']:
            if 'group_obs' in f:
                gen_obs.append(f['group_obs']) 
                gen_next_obs.append(f['group_next_obs'])
                gen_acts.append(f['group_acts'])
            if len(gen_obs)== self.old_demo_batch_size:
                break

        gen_samples['obs'] = np.vstack(gen_obs)
        gen_samples['acts'] = np.vstack(gen_acts).T[0]
        gen_samples['next_obs'] = np.vstack(gen_next_obs)

        #breakpoint()
        self.demo_batch_size = self.old_demo_batch_size * n_agents
        #TODO change obs, acts to be grouped togther

        return {'expert_samples':expert_samples,'gen_samples':gen_samples},f'{0} ** {0}'


    def train(
            self,
            total_timesteps: int,
            callback: Optional[Callable[[int], None]] = None,
        ) -> None:
            """Alternates between training the generator and discriminator.

            Every "round" consists of a call to `train_gen(self.gen_train_timesteps)`,
            a call to `train_disc`, and finally a call to `callback(round)`.

            Training ends once an additional "round" would cause the number of transitions
            sampled from the environment to exceed `total_timesteps`.

            Args:
                total_timesteps: An upper bound on the number of transitions to sample
                    from the environment during training.
                callback: A function called at the end of every round which takes in a
                    single argument, the round number. Round numbers are in
                    `range(total_timesteps // self.gen_train_timesteps)`.
            """
            n_rounds = total_timesteps // self.gen_train_timesteps
            assert n_rounds >= 1, (
                "No updates (need at least "
                f"{self.gen_train_timesteps} timesteps, have only "
                f"total_timesteps={total_timesteps})!"
            )
            for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
                self.train_gen(self.gen_train_timesteps)
                for _ in range(self.n_disc_updates_per_round):
                    with networks.training(self.reward_train):
                        # switch to training mode (affects dropout, normalization)
                        train_batch = self.prepare_for_disc()
                        self.train_disc(**train_batch)
                if callback:
                    callback(r)
                self.logger.dump(self._global_step)


