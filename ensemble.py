
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet,RewardNet
from imitation.util.networks import RunningNorm

from gym import spaces


def convert_batch(batch,n_agents=4,ma=False,device='cuda'):

    new_b = {}

    new_b.update({'state':batch["state"].reshape(-1,batch['state'].shape[1]//n_agents)})
    new_b.update({'next_state':batch["next_state"].reshape(-1,batch['next_state'].shape[1]//n_agents)})
    
    if ma:
        n_obs = batch["state"].reshape(-1,n_agents,batch['state'].shape[1]//n_agents)
        f_n_obs = torch.vstack([torch.Tensor(other_vars_4_ma(i_n_obs.cpu())).to(device) for i_n_obs in n_obs])
        new_b.update({'state':f_n_obs})

        n_obs = batch["next_state"].reshape(-1,n_agents,batch['next_state'].shape[1]//n_agents)
        f_n_obs = torch.vstack([torch.Tensor(other_vars_4_ma(i_n_obs.cpu())).to(device) for i_n_obs in n_obs])
        new_b.update({'next_state':f_n_obs})

    else:
        new_b.update({'state':batch["state"].reshape(-1,batch['state'].shape[1]//n_agents)})
        new_b.update({'next_state':batch["next_state"].reshape(-1,batch['next_state'].shape[1]//n_agents)})

    new_b.update({'action':batch["action"].reshape(-1,batch['action'].shape[1]//n_agents)})

    new_b.update({'done':torch.Tensor(
        np.array([batch["done"].cpu().numpy()]*n_agents).T.flatten()).to(device)})

    labels = batch["labels_expert_is_one"]

    new_b.update({'labels_expert_is_one':torch.Tensor(
        np.array([labels.cpu().numpy()]*n_agents).T.flatten()).to(device)})

    return new_b


def other_vars_4_ma(arr):

    res =[]

    for i  in range(arr.shape[0]):
        res.append(np.roll(arr,-i,axis=0))
    return np.array(res)

def from_many_to_single(roll_coop,n_agents=4,ma=False):

    # full list: N
        # rews : (traj,)
        # obs : (traj+1,obs_size)
        # acts : (traj,acts_size)
        # terminal : True
        # info : None

    roll_single = []
    steps = roll_coop[0].acts.shape[0]
    n_rews = np.zeros(steps)-1
    original_shape = list(roll_coop[0].obs.shape[1:]) #without steps
    original_shape[-1] = int(original_shape[-1]/n_agents)
    for traj in roll_coop:
        n_obs = traj.obs.reshape(steps+1,n_agents,*tuple(original_shape))
        n_acts = traj.acts.reshape(steps,n_agents,-1)

        if ma:
            f_n_obs = np.vstack([other_vars_4_ma(i_n_obs) for i_n_obs in n_obs])
        else:
            f_n_obs = n_obs.reshape((steps+1)*n_agents,*tuple(original_shape))
        for i in range(n_agents):

            if len(original_shape)>2:
                #quick fix
                obs = traj.obs[...,i*original_shape[-1]:(i+1)*original_shape[-1]]
            else:
                obs = f_n_obs[i::n_agents]
            acts = n_acts.reshape(steps*n_agents,-1)[i::n_agents]
            infos = []
            for s in range(steps):
                # for spread it was n_obs instead of traj.
                infos.append({
                    'group_obs':traj.obs[s,...].copy(),
                    'group_acts':traj.acts[s,...].copy(),
                    'group_next_obs':traj.obs[s+1,...].copy(),
                    'done':(s==(steps-1))
                    }
                )
            roll_single.append(
                TrajectoryWithRew(obs=obs,acts=acts.T[0],infos=infos,terminal=True,rews=n_rews)
            )

    return roll_single



def train_(DT,Di,labels,lr=0.01,epochs=2):

    """_summary_
    DT: (batch_size//n_agents,1)
    Di: (batch_size,1)
    labels : (batch_size//n_agents,1)
    """
    n_agents = len(Di)//len(DT)
    inputs = np.hstack((
        Di.reshape(-1,n_agents),DT.reshape(-1,1)))

    net = DsEnsemble()


    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i  in range(1):
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = F.binary_cross_entropy_with_logits(
                outputs,
                labels.float(),
            )

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')



class DsEnsemble(nn.Module):

    def __init__(self, n_agents=4,lr=0.01):
        super(DsEnsemble, self).__init__()
        # an affine operation: y = Wx + b
        self.n_rewards = 2
        #self.layers = nn.ModuleList([nn.Linear(self.n_rewards,1) for _ in range(n_agents)])
        self.n_agents = n_agents

        self.attend = nn.Linear(1+n_agents,1+n_agents)

        self.fc = nn.Linear(1+n_agents,n_agents)
        # No bais
        self.fc.bias.requires_grad = False

        # good initilization
        with torch.no_grad():
            self.fc.bias.fill_(0.)
            self.fc.weight.fill_(0.)
            for i in range(n_agents):
                self.fc.weight[i,i] = 0.9
                self.fc.weight[i,-1] = 0.1

        # create your optimizer
        self.optimizer = optim.Adam([self.fc.weight,
            self.attend.weight,self.attend.bias], lr=lr)

        #TODO print wights

    def forward(self, x):
        # input is 1+n_agents,1
        # Max pooling over a (2, 2) window

        #normalize

        #x -= 0.5

        att = self.attend(x)
        #no activation 

        x = torch.mul(att,x)
        x = F.relu(self.fc(x))

        return x

    def train_(self,DT,Di,labels,epochs=1,lr=0.01):

        """_summary_
        DT: (batch_size//n_agents,1)
        Di: (batch_size,1)
        labels : (batch_size//n_agents,1)

        input : Di, DT (last) [batch,n_agents]
        """
        n_agents = len(Di)//len(DT)
        inputs = torch.hstack((
            Di.reshape(-1,n_agents),DT.reshape(-1,1)))

        labels = labels.reshape(-1,n_agents)

        for g in self.optimizer.param_groups:
            g['lr'] = lr


        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i  in range(1):
                # get the inputs; data is a list of [inputs, labels]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                #print(inputs.shape)
                #print(outputs.shape)
                #print(labels.shape)
                loss = F.binary_cross_entropy_with_logits(
                    outputs,
                    labels.float(),
                )

                loss.backward()
                self.optimizer.step()

                print('Loss is: ', loss.item())
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')



class DsEnsemble2(nn.Module):

    """
    take global state and actions and return logits
    for the whole system
    """

    def __init__(self, obs_space,acts_space,n_agents=4,lr=0.01):
        super(DsEnsemble2, self).__init__()
        # an affine operation: y = Wx + b

        self.observation_space = obs_space
        self.action_space = acts_space
        self.normalize_images = True

        self.n_rewards = 2
        #self.layers = nn.ModuleList([nn.Linear(self.n_rewards,1) for _ in range(n_agents)])
        self.single_reward =  BasicRewardNet(
            obs_space, acts_space, use_next_state=False, normalize_input_layer=RunningNorm,
            activation=nn.Identity,#nn.LeakyReLU,
        )

        # it need -F.logsigmoid(-logits) for forward but not training
        self.group_obs = spaces.Box(
            float(obs_space.low_repr),float(obs_space.high_repr),(obs_space.shape[0]*n_agents,*obs_space.shape[1:])
            )

        self.group_acts = spaces.MultiDiscrete([acts_space.n]*n_agents)

        self.group_reward =  BasicRewardNet(
            self.group_obs, self.group_acts, use_next_state=False, normalize_input_layer=RunningNorm,
            activation=nn.ReLU,#nn.LeakyReLU,
        )

        self.n_agents = n_agents

        self.attend = nn.Linear(1+n_agents,1+n_agents)

        self.fc = nn.Linear(1+n_agents,n_agents)
        # No bais
        self.fc.bias.requires_grad = False

        # no good initilization
        if False:
            with torch.no_grad():
                self.fc.bias.fill_(0.)
                self.fc.weight.fill_(0.)
                for i in range(n_agents):
                    self.fc.weight[i,i] = 0.9
                    self.fc.weight[i,-1] = 0.1

        # create your optimizer
        self.optimizer = optim.Adam([self.fc.weight,
            self.attend.weight,self.attend.bias], lr=lr)

        self.single_optimizer = optim.Adam(
            self.single_reward.parameters(),
            lr=lr,
        )

        self.group_optimizer = optim.Adam(
            self.group_reward.parameters(),
            lr=lr,
        )

        #TODO print wights
    def parameters(self, recurse: bool = True):
        #return super().parameters(recurse)
        return [self.fc.weight,self.attend.weight,self.attend.bias]

    def forward(self,state, action, next_state, done):
        # input is 1+n_agents,1
        # Max pooling over a (2, 2) window

        with torch.no_grad():
            logit_s = self.single_reward(state, action, next_state, done)#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            logit_g = self.group_reward(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), next_state, done)#Batch
        #breakpoint()
        x = torch.hstack((logit_g.reshape(-1,1),logit_s.reshape(-1,self.n_agents))) # first row is the global (N+1,Batch)

        #normalize
        #x -= 0.5

        att = self.attend(x)
        #no activation 

        x = torch.mul(att,x)
        x = F.relu(self.fc(x))

        # it should be (batch,1)
        return x.flatten()

    def train_single(self,data_batch):

        self.single_reward.train()

        disc_logits = self.single_reward(
            data_batch["state"],
            data_batch["action"],
            data_batch["next_state"],
            data_batch["done"],
            #data_batch["log_policy_act_prob"],
        )

        loss = F.binary_cross_entropy_with_logits(
            disc_logits,
            data_batch["labels_expert_is_one"].float(),
        )

        # do gradient step
        self.single_optimizer.zero_grad()
        loss.backward()
        self.single_optimizer.step()

        self.single_reward.eval()
        #self._disc_step += 1

    def train_group(self,data_batch):


        self.group_reward.train()

        disc_logits = self.group_reward(
            data_batch["state"].reshape(-1,*self.group_obs.shape),
            data_batch["action"].reshape(-1,self.n_agents,self.action_space.n),
            data_batch["next_state"],
            data_batch["done"],
            #data_batch["log_policy_act_prob"],
        )

        loss = F.binary_cross_entropy_with_logits(
            disc_logits.repeat_interleave(self.n_agents,axis=0),
            data_batch["labels_expert_is_one"].float(),
        )

        # do gradient step
        self.group_optimizer.zero_grad()
        loss.backward()
        self.group_optimizer.step()

        self.group_reward.eval()
        #self._disc_step += 1

    def train_seprate(self,DT,Di,labels,epochs=1,lr=0.01):

        """_summary_
        DT: (batch_size//n_agents,1)
        Di: (batch_size,1)
        labels : (batch_size//n_agents,1)

        input : Di, DT (last) [batch,n_agents]
        """
        n_agents = len(Di)//len(DT)
        inputs = torch.hstack((
            Di.reshape(-1,n_agents),DT.reshape(-1,1)))

        labels = labels.reshape(-1,n_agents)

        for g in self.optimizer.param_groups:
            g['lr'] = lr


        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i  in range(1):
                # get the inputs; data is a list of [inputs, labels]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                #print(inputs.shape)
                #print(outputs.shape)
                #print(labels.shape)
                loss = F.binary_cross_entropy_with_logits(
                    outputs,
                    labels.float(),
                )

                loss.backward()
                self.optimizer.step()

                print('Loss is: ', loss.item())
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')


class DsEnsemble3(DsEnsemble2):

    """
    self-attention
    take global state and actions and return logits
    for the whole system
    """

    def __init__(self, obs_space,acts_space,n_agents=4,lr=0.01):

        super(DsEnsemble3, self).__init__( obs_space,acts_space,n_agents=n_agents,lr=lr)
        self.attend = nn.Linear(1+n_agents,(1+n_agents)**2)


    def forward(self,state, action, next_state, done):
        # input is 1+n_agents,1
        # Max pooling over a (2, 2) window

        with torch.no_grad():
            logit_s = self.single_reward(state, action, next_state, done)#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            logit_g = self.group_reward(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), next_state, done)#Batch

            #breakpoint()
            #logit_g = torch.sigmoid(logit_g)
            #logit_g = torch.clip(logit_g ,torch.Tensor([0.00001]).cuda(),torch.Tensor([0.9999]).cuda())
            #logit_s = torch.clip(logit_s ,torch.Tensor([0.00001]).cuda(),torch.Tensor([0.9999]).cuda())
            #logit_s = torch.sigmoid(logit_s)
            #r_g = torch.mul(logit_g-0.5,10)#torch.log(logit_g)-torch.log(1-logit_g)#-F.logsigmoid(-logit_g)
            r_g = torch.log(logit_g+1e-7)
            #r_s = torch.mul(logit_s-0.5,10)#torch.log(logit_s)-torch.log(1-logit_s)#-F.logsigmoid(-logit_s)
            r_s = torch.log(logit_s+1e-7)

        x = torch.hstack((r_g.reshape(-1,1),r_s.reshape(-1,self.n_agents))) 
        # first row is the global (N+1,Batch)

        #normalize
        #x -= 0.5

        att = self.attend(x)
        #no activation 
        # TODO  check the values
        x = torch.mul(att,x.repeat(1,1+self.n_agents))# for batch
        x = x.reshape(-1,1+self.n_agents,1+self.n_agents).sum(axis=2)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        #x = torch.clip(x,torch.Tensor([0.00001]).cuda(),torch.Tensor([0.9999]).cuda())
        #x = F.leaky_relu(self.fc(x), negative_slope=0.01) # to allow negative rewards

        # it should be (batch,1)
        return x.flatten()
    


class DsEnsemble4(DsEnsemble2):

    """
    self-attention
    take global state and actions and return logits
    for the whole system
    """

    def __init__(self, obs_space,acts_space,n_agents=4,lr=0.01):

        super(DsEnsemble4, self).__init__( obs_space,acts_space,n_agents=n_agents,lr=lr)
        in_size = spaces.utils.flatdim(self.group_acts)+spaces.utils.flatdim(self.group_obs)
        self.f_s_a = nn.Linear(in_size,n_agents)  #(S,A)==> softmax(N) for N agent DT

        #TODO print wights
    def parameters(self, recurse: bool = True):
        #return super().parameters(recurse)
        return [self.f_s_a.weight,self.f_s_a.bias]

    def forward(self,state, action, next_state, done):
        # input is 1+n_agents,1
        # Max pooling over a (2, 2) window

        with torch.no_grad():
            logit_s = self.single_reward(state, action, next_state, done)#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            logit_g = self.group_reward(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), next_state, done)#Batch

            #breakpoint()
            #logit_g = torch.sigmoid(logit_g)
            #logit_g = torch.clip(logit_g ,torch.Tensor([0.00001]).cuda(),torch.Tensor([0.9999]).cuda())
            #logit_s = torch.clip(logit_s ,torch.Tensor([0.00001]).cuda(),torch.Tensor([0.9999]).cuda())
            #logit_s = torch.sigmoid(logit_s)
            #r_g = torch.log(logit_g+1e-7)#-F.logsigmoid(-logit_g)#
            r_g = logit_g#-F.logsigmoid(-logit_g)
            #r_s = torch.mul(logit_s-0.5,10)#torch.log(logit_s)-torch.log(1-logit_s)#-F.logsigmoid(-logit_s)
            #r_s = torch.log(logit_s+1e-7)#-F.logsigmoid(-logit_s)
            r_s = logit_s#-F.logsigmoid(-logit_s)


        #breakpoint()
        full_states = torch.hstack((state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents*self.action_space.n)))

        centralized_w = F.softmax(self.f_s_a(full_states),dim=1)#24,3
        decentralized_w = 1 - centralized_w
        g_factor = r_g.repeat(self.n_agents,1).T.flatten()*(centralized_w.flatten())
        l_factor = r_s* (decentralized_w.flatten())
        
        x = (g_factor+l_factor)#.reshape(-1,self.n_agents) # 24
        #x = torch.sigmoid(x)
        #x = torch.clip(x,torch.Tensor([0.00001]).cuda(),torch.Tensor([0.9999]).cuda())
        #x = F.leaky_relu(self.fc(x), negative_slope=0.01) # to allow negative rewards

        # it should be (batch,1)
        #y =torch.logit(torch.exp(-1*x))
        #if  torch.isnan(y).any():
        #    breakpoint()
        return x#-1*torch.logit(torch.exp(-1*x))#.flatten()
    


class Shaped_R_Blended2(RewardNet):

    """
    take global state and actions and return logits
    for the whole system
    """

    def __init__(self, obs_space,acts_space,n_agents=4,lr=0.01):
        super(Shaped_R_Blended2, self).__init__(observation_space=obs_space,action_space=acts_space)
        # an affine operation: y = Wx + b

        self.observation_space = obs_space
        self.action_space = acts_space
        self.normalize_images = True

        self.rew_range = 50
        self.n_rewards = 2
        #self.layers = nn.ModuleList([nn.Linear(self.n_rewards,1) for _ in range(n_agents)])
        self.single_reward =  BasicShapedRewardNet(
            obs_space, acts_space, use_next_state=False, normalize_input_layer=RunningNorm,
            activation=nn.LeakyReLU,#nn.LeakyReLU,
        )
        self.group_obs = spaces.Box(
            float(obs_space.low_repr),float(obs_space.high_repr),(obs_space.shape[0]*n_agents,*obs_space.shape[1:])
            )

        self.group_acts = spaces.MultiDiscrete([acts_space.n]*n_agents)

        #self.group_obs = spaces.Box(
        #    (obs_space.low_repr),(obs_space.high_repr),(*obs_space.shape[:-1],obs_space.shape[-1]*n_agents)
        #    )

        #self.group_acts = spaces.MultiDiscrete([acts_space.n]*n_agents)

        self.group_reward =  BasicShapedRewardNet(
            self.group_obs, self.group_acts, use_next_state=False, normalize_input_layer=RunningNorm,
            activation=nn.LeakyReLU,#nn.LeakyReLU,
        )

        self.n_agents = n_agents

        in_size = spaces.utils.flatdim(self.group_acts)+spaces.utils.flatdim(self.group_obs)
        self.f_s_a = nn.Sequential(nn.Linear(in_size,32),
                                   nn.Identity(),
                                   nn.Linear(32,n_agents),
                                   nn.Identity(),
                                   nn.Sigmoid(),
                                   #nn.ReLU(),
                                   #nn.Softmax(dim=1),
                                   )  #(S,A)==> softmax(N) for N agent DT

        # no good initilization
        if False:
            with torch.no_grad():
                self.fc.bias.fill_(0.)
                self.fc.weight.fill_(0.)
                for i in range(n_agents):
                    self.fc.weight[i,i] = 0.9
                    self.fc.weight[i,-1] = 0.1

        # create your optimizer

        self.single_optimizer = optim.Adam(
            self.single_reward.parameters(),
            lr=lr,
        )

        self.group_optimizer = optim.Adam(
            self.group_reward.parameters(),
            lr=lr,
        )

        self.group_trained = 0
        self.single_trained = 0

        #TODO print wights

    def init_f_s_a(self,mode=0):
        #0 : full
        #1 : decent
        #2 : cent
        if mode in [1,2]:
            #final layer all zeros
            self.f_s_a = self.f_s_a[:-1] #delet softmax
            self.f_s_a[2].weight.data.fill_(0.0)
            self.f_s_a[2].bias.data.fill_((mode-1)/self.n_agents)

            for param in self.f_s_a.parameters():
                param.requires_grad = False


    def parameters(self, recurse: bool = True):
        #return super().parameters(recurse)
        return self.f_s_a.parameters()

    def forward(self,state, action, next_state, done):
        # input is 1+n_agents,1
        # Max pooling over a (2, 2) window
        full_states = torch.hstack((state.detach().clone().reshape(-1,np.prod(self.group_obs.shape)),
                        action.detach().clone().reshape(-1,self.n_agents*self.action_space.n)))

        with torch.no_grad():
            shaped_r_s = self.single_reward(state, action, next_state, done.flatten())#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            shaped_r_s = torch.clip(shaped_r_s, min=-self.rew_range, max=self.rew_range)
            shaped_r_g = self.group_reward(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), 
                        next_state.reshape(-1,*self.group_obs.shape), 
                        done.reshape(-1,self.n_agents).all(axis=1))#Batch
            shaped_r_g = torch.clip(shaped_r_g, min=-self.rew_range, max=self.rew_range)


        if self.group_trained==self.single_trained:
            # normlize r_s to r_g
            if shaped_r_s.max():
                shaped_r_s = (shaped_r_s/shaped_r_s.max())*(shaped_r_g.max()+0.01)
            centralized_w = self.f_s_a(full_states)#24,3
            #centralized_w = torch.div(centralized_w.T,torch.abs(centralized_w).sum(axis=1)).T
            decentralized_w = 1 - centralized_w
            g_factor = shaped_r_g.repeat(self.n_agents,1).T.flatten()*(centralized_w.flatten())
            l_factor = shaped_r_s* (decentralized_w.flatten())
            
            x = (g_factor+l_factor)#.reshape(-1,self.n_agents) # 24
        elif self.group_trained:
            x = shaped_r_g.repeat(self.n_agents,1).T.flatten()
        elif self.single_trained:
            x = shaped_r_s

        # it should be (batch,1)
        return x

    def train_single(self,data_batch):

        self.single_reward.train()

        rew = self.single_reward(
            data_batch["state"].detach().clone(),
            data_batch["action"].detach().clone(),
            data_batch["next_state"].detach().clone(),
            data_batch["done"].detach().clone().flatten(),
        )
        rew = torch.clip(rew, min=-self.rew_range, max=self.rew_range)
        #low_exp = torch.exp(rew-data_batch["log_policy_act_prob"].detach().clone())
        #disc_logits = 1/(1+low_exp)
        disc_logits = torch.exp(rew)/(
            torch.exp(rew)+torch.exp(data_batch["log_policy_act_prob"].detach().clone()))

        #disc_logits = rew -data_batch["log_policy_act_prob"].detach().clone()

        loss = F.binary_cross_entropy(#_with_logits(
            disc_logits,
            data_batch["labels_expert_is_one"].detach().clone().float(),
        )

        # do gradient step
        self.single_optimizer.zero_grad()
        loss.backward()
        self.single_optimizer.step()

        self.single_reward.eval()
        self.single_trained = 1
        #self._disc_step += 1
    def get_reward_and_potional(self,network,state, action, next_state, done):
        reward = network.base(state, action, next_state, done)

    def train_group(self,data_batch):

        self.group_reward.train()
        #breakpoint()
        rew = self.group_reward(
            data_batch["state"].detach().clone().reshape(-1,*self.group_obs.shape),
            data_batch["action"].detach().clone().reshape(-1,self.n_agents,self.action_space.n),
            data_batch["next_state"].detach().clone().reshape(-1,*self.group_obs.shape),
            data_batch["done"].detach().clone().reshape(-1,self.n_agents).all(axis=1),
            
        )
        rew = torch.clip(rew, min=-self.rew_range, max=self.rew_range)
        #disc_logits = rew - -(data_batch["log_policy_act_prob"].detach(
        #        ).clone().reshape(-1,self.n_agents).sum(axis=1))
        #low_exp = torch.exp(rew-(data_batch["log_policy_act_prob"].detach(
        #        ).clone().reshape(-1,self.n_agents).sum(axis=1)))
        #disc_logits = 1/(1+low_exp)
        disc_logits = torch.exp(rew)/(
            torch.exp(rew)+torch.exp(data_batch["log_policy_act_prob"].detach(
                ).clone().reshape(-1,self.n_agents).sum(axis=1)))

        loss = F.binary_cross_entropy(#_with_logits(
            disc_logits.repeat_interleave(self.n_agents,axis=0),
            data_batch["labels_expert_is_one"].detach().clone().float(),
        )

        # do gradient step
        self.group_optimizer.zero_grad()
        loss.backward()
        self.group_optimizer.step()

        self.group_reward.eval()
        #self._disc_step += 1
        self.group_trained = 1

    def new_reward(self,state_, action_, next_state_, done_):

        state, action, next_state, done = self.preprocess(
            state_,
            action_,
            next_state_,
            done_,
        )


        full_states = torch.hstack((state.detach().clone().reshape(-1,np.prod(self.group_obs.shape)),
                        action.detach().clone().reshape(-1,self.n_agents*self.action_space.n)))

        with torch.no_grad():
            r_s = self.single_reward.base(state, action, next_state, done.flatten())#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            r_s = torch.clip(r_s, min=-self.rew_range, max=self.rew_range)
            r_g = self.group_reward.base(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), 
                        next_state.reshape(-1,*self.group_obs.shape), 
                        done.reshape(-1,self.n_agents).all(axis=1))#Batch
            r_g = torch.clip(r_g, min=-self.rew_range, max=self.rew_range)


            if  self.group_trained==self.single_trained:
                # normlize r_s to r_g
                if r_s.max():
                    r_s = (r_s/r_s.max())*(r_g.max()+0.01)
                centralized_w = self.f_s_a(full_states)#24,3
                #centralized_w = torch.div(centralized_w.T,torch.abs(centralized_w).sum(axis=1)).T

                decentralized_w = 1 - centralized_w
                g_factor = r_g.repeat(self.n_agents,1).T.flatten()*(centralized_w.flatten())
                l_factor = r_s* (decentralized_w.flatten())
                
                x = (g_factor+l_factor)#.reshape(-1,self.n_agents) # 24
            elif self.group_trained:
                x = r_g.repeat(self.n_agents,1).T.flatten()
            elif self.single_trained:
                x = r_s
        #breakpoint()
        # it should be (batch,1)
        return x.detach().cpu().numpy().flatten()
    

class Shaped_R_Blended(RewardNet):

    """
    take global state and actions and return logits
    for the whole system
    """

    def __init__(self, obs_space,acts_space,n_agents=4,lr=0.01):
        super(Shaped_R_Blended, self).__init__(observation_space=obs_space,action_space=acts_space)
        # an affine operation: y = Wx + b

        self.observation_space = obs_space
        self.action_space = acts_space
        self.normalize_images = True

        self.rew_range = 50
        self.n_rewards = 2
        #self.layers = nn.ModuleList([nn.Linear(self.n_rewards,1) for _ in range(n_agents)])
        self.single_reward =  BasicShapedRewardNet(
            obs_space, acts_space, use_next_state=False, normalize_input_layer=RunningNorm,
            activation=nn.ReLU,#nn.LeakyReLU,
        )
        self.group_obs = spaces.Box(
            float(obs_space.low_repr),float(obs_space.high_repr),(obs_space.shape[0]*n_agents,*obs_space.shape[1:])
            )

        self.group_acts = spaces.MultiDiscrete([acts_space.n]*n_agents)

        #self.group_obs = spaces.Box(
        #    (obs_space.low_repr),(obs_space.high_repr),(*obs_space.shape[:-1],obs_space.shape[-1]*n_agents)
        #    )

        #self.group_acts = spaces.MultiDiscrete([acts_space.n]*n_agents)

        self.group_reward =  BasicShapedRewardNet(
            self.group_obs, self.group_acts, use_next_state=False, normalize_input_layer=RunningNorm,
            activation=nn.ReLU,#nn.LeakyReLU,
        )

        self.n_agents = n_agents

        in_size = spaces.utils.flatdim(self.group_acts)+spaces.utils.flatdim(self.group_obs)
        self.f_s_a = nn.Sequential(nn.Linear(in_size,32),
                                   nn.ReLU(),
                                   nn.Linear(32,n_agents),
                                   nn.ReLU(),
                                   nn.Softmax(dim=1),
                                   )  #(S,A)==> softmax(N) for N agent DT

        # no good initilization
        if False:
            with torch.no_grad():
                self.fc.bias.fill_(0.)
                self.fc.weight.fill_(0.)
                for i in range(n_agents):
                    self.fc.weight[i,i] = 0.9
                    self.fc.weight[i,-1] = 0.1

        # create your optimizer

        self.single_optimizer = optim.Adam(
            self.single_reward.parameters(),
            lr=lr,
        )

        self.group_optimizer = optim.Adam(
            self.group_reward.parameters(),
            lr=lr,
        )

        self.group_trained = 0
        self.single_trained = 0

        #TODO print wights

    def init_f_s_a(self,mode=0):
        #0 : full
        #1 : decent
        #2 : cent
        if mode in [1,2]:
            #final layer all zeros
            self.f_s_a = self.f_s_a[:-1] #delet softmax
            self.f_s_a[2].weight.data.fill_(0.0)
            self.f_s_a[2].bias.data.fill_((mode-1)/self.n_agents)

            for param in self.f_s_a.parameters():
                param.requires_grad = False


    def parameters(self, recurse: bool = True):
        #return super().parameters(recurse)
        return self.f_s_a.parameters()

    def forward(self,state, action, next_state, done):
        # input is 1+n_agents,1
        # Max pooling over a (2, 2) window
        full_states = torch.hstack((state.detach().clone().reshape(-1,np.prod(self.group_obs.shape)),
                        action.detach().clone().reshape(-1,self.n_agents*self.action_space.n)))

        with torch.no_grad():
            shaped_r_s = self.single_reward(state, action, next_state, done.flatten())#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            shaped_r_s = torch.clip(shaped_r_s, min=-self.rew_range, max=self.rew_range)
            shaped_r_g = self.group_reward(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), 
                        next_state.reshape(-1,*self.group_obs.shape), 
                        done.reshape(-1,self.n_agents).all(axis=1))#Batch
            shaped_r_g = torch.clip(shaped_r_g, min=-self.rew_range, max=self.rew_range)


        if self.group_trained==self.single_trained:
            centralized_w = self.f_s_a(full_states)*self.n_agents#24,3
            #centralized_w = torch.div(centralized_w.T,torch.abs(centralized_w).sum(axis=1)).T
            decentralized_w = 1 - centralized_w
            g_factor = shaped_r_g.repeat(self.n_agents,1).T.flatten()*(centralized_w.flatten())
            l_factor = shaped_r_s* (decentralized_w.flatten())
            
            x = (g_factor+l_factor)#.reshape(-1,self.n_agents) # 24
        elif self.group_trained:
            x = shaped_r_g.repeat(self.n_agents,1).T.flatten()
        elif self.single_trained:
            x = shaped_r_s

        # it should be (batch,1)
        return x

    def train_single(self,data_batch):

        self.single_reward.train()

        rew = self.single_reward(
            data_batch["state"].detach().clone(),
            data_batch["action"].detach().clone(),
            data_batch["next_state"].detach().clone(),
            data_batch["done"].detach().clone().flatten(),
        )
        rew = torch.clip(rew, min=-self.rew_range, max=self.rew_range)
        #low_exp = torch.exp(rew-data_batch["log_policy_act_prob"].detach().clone())
        #disc_logits = 1/(1+low_exp)
        disc_logits = torch.exp(rew)/(
            torch.exp(rew)+torch.exp(data_batch["log_policy_act_prob"].detach().clone()))

        #disc_logits = rew -data_batch["log_policy_act_prob"].detach().clone()

        loss = F.binary_cross_entropy(#_with_logits(
            disc_logits,
            data_batch["labels_expert_is_one"].detach().clone().float(),
        )

        # do gradient step
        self.single_optimizer.zero_grad()
        loss.backward()
        self.single_optimizer.step()

        self.single_reward.eval()
        self.single_trained = 1
        #self._disc_step += 1
    def get_reward_and_potional(self,network,state, action, next_state, done):
        reward = network.base(state, action, next_state, done)

    def train_group(self,data_batch):

        self.group_reward.train()
        #breakpoint()
        rew = self.group_reward(
            data_batch["state"].detach().clone().reshape(-1,*self.group_obs.shape),
            data_batch["action"].detach().clone().reshape(-1,self.n_agents,self.action_space.n),
            data_batch["next_state"].detach().clone().reshape(-1,*self.group_obs.shape),
            data_batch["done"].detach().clone().reshape(-1,self.n_agents).all(axis=1),
            
        )
        rew = torch.clip(rew, min=-self.rew_range, max=self.rew_range)
        #disc_logits = rew - -(data_batch["log_policy_act_prob"].detach(
        #        ).clone().reshape(-1,self.n_agents).sum(axis=1))
        #low_exp = torch.exp(rew-(data_batch["log_policy_act_prob"].detach(
        #        ).clone().reshape(-1,self.n_agents).sum(axis=1)))
        #disc_logits = 1/(1+low_exp)
        disc_logits = torch.exp(rew)/(
            torch.exp(rew)+torch.exp(data_batch["log_policy_act_prob"].detach(
                ).clone().reshape(-1,self.n_agents).sum(axis=1)))

        loss = F.binary_cross_entropy(#_with_logits(
            disc_logits.repeat_interleave(self.n_agents,axis=0),
            data_batch["labels_expert_is_one"].detach().clone().float(),
        )

        # do gradient step
        self.group_optimizer.zero_grad()
        loss.backward()
        self.group_optimizer.step()

        self.group_reward.eval()
        #self._disc_step += 1
        self.group_trained = 1

    def new_reward(self,state_, action_, next_state_, done_):

        state, action, next_state, done = self.preprocess(
            state_,
            action_,
            next_state_,
            done_,
        )


        full_states = torch.hstack((state.detach().clone().reshape(-1,np.prod(self.group_obs.shape)),
                        action.detach().clone().reshape(-1,self.n_agents*self.action_space.n)))

        with torch.no_grad():
            r_s = self.single_reward.base(state, action, next_state, done.flatten())#N*Batch
            # old_bacth ==> (first dim batches, second n_agents), third obs_shape
            r_s = torch.clip(r_s, min=-self.rew_range, max=self.rew_range)
            r_g = self.group_reward.base(state.reshape(-1,*self.group_obs.shape),
                        action.reshape(-1,self.n_agents,self.action_space.n), 
                        next_state.reshape(-1,*self.group_obs.shape), 
                        done.reshape(-1,self.n_agents).all(axis=1))#Batch
            r_g = torch.clip(r_g, min=-self.rew_range, max=self.rew_range)


            if  self.group_trained==self.single_trained:
                centralized_w = self.f_s_a(full_states)*self.n_agents#24,3
                #centralized_w = torch.div(centralized_w.T,torch.abs(centralized_w).sum(axis=1)).T

                decentralized_w = 1 - centralized_w
                g_factor = r_g.repeat(self.n_agents,1).T.flatten()*(centralized_w.flatten())
                l_factor = r_s* (decentralized_w.flatten())
                
                x = (g_factor+l_factor)#.reshape(-1,self.n_agents) # 24
            elif self.group_trained:
                x = r_g.repeat(self.n_agents,1).T.flatten()
            elif self.single_trained:
                x = r_s
        #breakpoint()
        # it should be (batch,1)
        return x.detach().cpu().numpy().flatten()