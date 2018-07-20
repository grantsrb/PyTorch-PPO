import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

def cuda_if(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj

class Updater():
    """
    This class converts the data collected from the rollouts into useable data to update
    the model. The main function to use is calc_loss which accepts a rollout to
    add to the global loss of the model. The model isn't updated, however, until calling
    calc_gradients followed by update_model. If the size of the epoch is restricted by the memory, you can call calc_gradients to clear the graph.
    """

    def __init__(self, net, hyps): 
        self.net = net 
        self.old_net = copy.deepcopy(self.net) 
        self.hyps = hyps
        self.entr_coef = hyps['entr_coef']
        self.val_coef = hyps['val_coef']
        self.gamma = hyps['gamma']
        self.lambda_ = hyps['lambda_']
        self.max_norm = hyps['max_norm']
        self.n_epochs = hyps['n_epochs']
        self.batch_size = hyps['batch_size']
        self.epsilon = hyps['epsilon']
        self.clip_vals = hyps['clip_vals']
        self.norm_advs = hyps['norm_advs']
        self.norm_batch_advs = hyps['norm_batch_advs']
        self.use_nstep_rets = hyps['use_nstep_rets']
        self.optim_type = hyps['optim_type']
        self.optim = self.new_optim(hyps['lr'])    

        # Tracking variables
        self.info = {}
        self.max_adv = -1
        self.min_adv = 1
        self.max_minsurr = -1e10
        self.min_minsurr = 1e10

    def update_model(self, shared_data):
        """
        This function accepts the data collected from a rollout and performs PPO update iterations
        on the neural net.

        datas - dict of torch tensors with shared memory to collect data. Each 
                tensor contains indices from idx*n_tsteps to (idx+1)*n_tsteps
                Keys (assume string keys):
                    "states" - MDP states at each timestep t
                            type: FloatTensor
                            shape: (n_states, *state_shape)
                    "h_states" - recurrent states at each timestep
                            type: FloatTensor
                            shape: (n_states, *state_shape)
                    "rewards" - Collects float rewards collected at each timestep t
                            type: FloatTensor
                            shape: (n_states,)
                    "dones" - Collects the dones collected at each timestep t
                            type: FloatTensor
                            shape: (n_states,)
                    "actions" - Collects actions performed at each timestep t
                            type: LongTensor
                            shape: (n_states,)
                    "deltas" - the Bellman differentials collected during the rollout
                            type: FloatTensor
                            shape: (n_states,)
        """

        hyps = self.hyps

        states = shared_data['states']
        rewards = shared_data['rewards']
        dones = shared_data['dones']
        actions = shared_data['actions']
        deltas = shared_data['deltas']

        # Make Advs and Returns
        advs = cuda_if(self.discount(deltas.squeeze(), dones.squeeze(), hyps['gamma']*hyps['lambda_']))
        if self.use_nstep_rets: 
            self.net.req_grads(False)
            if self.net.is_recurrent:
                hs = shared_data['h_states']
                vals, pis, _ = self.net(Variable(states), Variable(hs))
            else:
                vals, pis = self.net(Variable(states))
            returns = advs + vals.data.squeeze()
        else: 
            returns = cuda_if(self.discount(rewards.squeeze(), dones.squeeze(), self.gamma))

        if self.norm_advs:
            advs = (advs - advs.mean())/(advs.std()+1e-5)

        avg_epoch_loss,avg_epoch_policy_loss,avg_epoch_val_loss,avg_epoch_entropy = 0,0,0,0
        self.net.train(mode=True)
        self.net.req_grads(True)
        self.old_net.load_state_dict(self.net.state_dict())
        self.old_net.train(mode=True)
        self.old_net.req_grads(False)
        self.optim.zero_grad()
        for epoch in range(self.n_epochs):

            loss, epoch_loss, epoch_policy_loss, epoch_val_loss, epoch_entropy = 0,0,0,0,0
            indices = cuda_if(torch.randperm(len(states)).long())

            for i in range(len(indices)//self.batch_size):

                # Get data for batch
                startdx = i*self.batch_size
                endx = (i+1)*self.batch_size
                idxs = indices[startdx:endx]
                if self.net.is_recurrent:
                    hs = shared_data['h_states']
                    batch_data = states[idxs],actions[idxs],advs[idxs],returns[idxs],hs[idxs] 
                else:
                    batch_data = states[idxs],actions[idxs],advs[idxs],returns[idxs],None

                # Total Loss
                policy_loss, val_loss, entropy = self.ppo_losses(*batch_data)
                loss = policy_loss + val_loss - entropy

                # Gradient Step
                loss.backward()
                self.norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
                self.optim.step()
                self.optim.zero_grad()
                epoch_loss += float(loss.data)
                epoch_policy_loss += float(policy_loss.data)
                epoch_val_loss += float(val_loss.data)
                epoch_entropy += float(entropy.data)

            avg_epoch_loss += epoch_loss/self.n_epochs
            avg_epoch_policy_loss += epoch_policy_loss/self.n_epochs
            avg_epoch_val_loss += epoch_val_loss/self.n_epochs
            avg_epoch_entropy += epoch_entropy/self.n_epochs

        self.info = {"Loss":float(avg_epoch_loss), 
                    "PiLoss":float(avg_epoch_policy_loss), 
                    "VLoss":float(avg_epoch_val_loss), 
                    "S":float(avg_epoch_entropy), 
                    "MaxAdv":float(self.max_adv),
                    "MinAdv":float(self.min_adv), 
                    "MinSurr":float(self.min_minsurr), 
                    "MaxSurr":float(self.max_minsurr)} 
        self.max_adv, self.min_adv, = -1, 1
        self.max_minsurr, self.min_minsurr = -1e10, 1e10

    def ppo_losses(self, states, actions, advs, rets, hs=None):
        """
        Performs the ppo specific loss approach

        states - torch FloatTensor minibatch of states with shape (batch_size, C, H, W)
        actions - torch LongTensor minibatch of empirical actions with shape (batch_size,)
        advs - torch FloatTensor minibatch of empirical advantages with shape (batch_size,)
        rets - torch FloatTensor minibatch of empirical returns with shape (batch_size,)
        hs - torch FloatTensor minibatch of recurrent states with shape (batch_size,h_size)

        Returns:
            policy_loss - the PPO CLIP policy gradient shape (1,)
            val_loss - the critic loss shape (1,)
            entropy - the entropy of the action predictions shape (1,)
        """
        # Get Outputs
        if hs is not None:
            vals, raw_pis, _ = self.net(Variable(states), Variable(hs))
            old_vals, old_raw_pis, _ = self.old_net(Variable(states), Variable(hs))
        else:
            vals, raw_pis = self.net(Variable(states))
            old_vals, old_raw_pis = self.old_net(Variable(states))

        old_vals.detach(), old_raw_pis.detach()
        probs = F.softmax(raw_pis, dim=-1)
        pis = probs[cuda_if(torch.arange(0,len(probs)).long()), actions]
        old_probs = F.softmax(old_raw_pis, dim=-1)
        old_pis = old_probs[cuda_if(torch.arange(0,len(old_probs))).long(), actions]

        # Policy Loss
        if self.norm_batch_advs:
            advs = (advs - advs.mean())
            advs = advs / (advs.std() + 1e-7)
        self.max_adv = max(torch.max(advs), self.max_adv) # Tracking variable
        self.min_adv = min(torch.min(advs), self.min_adv) # Tracking variable

        advs = Variable(advs)
        ratio = pis/(old_pis+1e-5)
        surrogate1 = ratio*advs
        surrogate2 = torch.clamp(ratio, 1.-self.epsilon, 1.+self.epsilon)*advs
        min_surr = torch.min(surrogate1, surrogate2)
        self.max_minsurr = max(torch.max(min_surr.data), self.max_minsurr)
        self.min_minsurr = min(torch.min(min_surr.data), self.min_minsurr)
        policy_loss = -min_surr.mean()

        # Value loss
        rets = Variable(rets)
        if self.clip_vals:
            clipped_vals = old_vals + torch.clamp(vals-old_vals, -self.epsilon, self.epsilon)
            v1 = .5*(vals.squeeze()-rets)**2
            v2 = .5*(clipped_vals.squeeze()-rets)**2
            val_loss = self.val_coef * torch.max(v1,v2).mean()
        else:
            val_loss = self.val_coef * F.mse_loss(vals.squeeze(), rets)

        # Entropy Loss
        softlogs = F.log_softmax(raw_pis, dim=-1)
        entropy_step = torch.sum(softlogs*probs, dim=-1)
        entropy = -self.entr_coef * entropy_step.mean()

        return policy_loss, val_loss, entropy

    def discount(self, array, dones, discount_factor):
        """
        Dicounts the argued array following the bellman equation.

        array - array to be discounted
        dones - binary array denoting the end of an episode
        discount_factor - float between 0 and 1 used to discount the reward

        Returns the discounted array as a torch FloatTensor
        """
        running_sum = 0
        discounts = cuda_if(torch.zeros(len(array)))
        for i in reversed(range(len(array))):
            if dones[i] == 1: running_sum = 0
            running_sum = array[i] + discount_factor*running_sum
            discounts[i] = running_sum
        return discounts

    def print_statistics(self):
        print(" – ".join([key+": "+str(round(val,5)) for key,val in sorted(self.info.items())]))

    def log_statistics(self, log, T, reward, avg_action, best_avg_rew):
        log.write("Step:"+str(T)+" – "+" – ".join([key+": "+str(round(val,5)) if "ntropy" not in key else key+": "+str(val) for key,val in self.info.items()]+["EpRew: "+str(reward), "AvgAction: "+str(avg_action), "BestRew:"+str(best_avg_rew)]) + '\n')
        log.flush()

    def save_model(self, net_file_name, optim_file_name):
        """
        Saves the state dict of the model to file.

        file_name - string name of the file to save the state_dict to
        """
        torch.save(self.net.state_dict(), net_file_name)
        if optim_file_name is not None:
            torch.save(self.optim.state_dict(), optim_file_name)
    
    def new_lr(self, new_lr):
        new_optim = self.new_optim(new_lr)
        new_optim.load_state_dict(self.optim.state_dict())
        self.optim = new_optim

    def new_optim(self, lr):
        if self.optim_type == 'rmsprop':
            new_optim = optim.RMSprop(self.net.parameters(), lr=lr) 
        elif self.optim_type == 'adam':
            new_optim = optim.Adam(self.net.parameters(), lr=lr) 
        else:
            new_optim = optim.RMSprop(self.net.parameters(), lr=lr) 
        return new_optim
