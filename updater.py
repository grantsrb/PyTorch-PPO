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

    def __init__(self, net, lr, entropy_const=0.01, value_const=0.5, gamma=0.99, lambda_=0.98, max_norm=0.5, n_epochs=3, batch_size=128, cache_size=0, epsilon=.2, fresh_advs=False, clip_vals=False, norm_returns=False, norm_advs=True, norm_batch_advs=False, eval_vals=True, use_nstep_rets=True):
        self.net = net
        self.old_net = copy.deepcopy(self.net)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.entropy_const = entropy_const
        self.val_const = value_const
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_norm = max_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.epsilon = epsilon
        self.fresh_advs = fresh_advs
        self.clip_vals = clip_vals
        self.norm_returns = norm_returns
        self.norm_advs = norm_advs
        self.norm_batch_advs = norm_batch_advs
        self.eval_vals = eval_vals
        self.use_nstep_rets = use_nstep_rets

        # Data caches
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.advantages = []

        # Tracking variables
        self.info = {}
        self.max_adv = -1
        self.min_adv = 1
        self.max_ret = -1
        self.min_ret = 1
        self.avg_disc_rew = 0

        if torch.cuda.is_available():
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor

    def update_model(self, states, next_states, rewards, dones, actions):
        """
        This function accepts the data collected from a rollout and performs PPO update iterations
        on the neural net.

        states - python list of the environment states from the rollouts
                shape = (n_states, *state_shape)
        next_states - python list of the next environment states from the rollouts
                shape = (n_states, *state_shape)
        rewards - python list of rewards from the rollouts
                shape = (n_states,)
        dones - python list of done signals from the rollouts
                dones = (n_states,)
        actions - python integer list denoting the actual action
                indexes taken in the rollout
        """

        if self.cache_size is not None and self.cache_size > 0:
            self.states,self.next_states,self.rewards,self.actions,self.dones =self.states+states,\
                                                                self.next_states+next_states,\
                                                                self.rewards+rewards,\
                                                                self.actions+actions,\
                                                                self.dones+dones
            self.states = self.states[-self.cache_size:]
            self.next_states = self.next_states[-self.cache_size:]
            self.rewards = self.rewards[-self.cache_size:]
            self.actions = self.actions[-self.cache_size:]
            self.dones = self.dones[-self.cache_size:]
        else:
            self.states, self.next_states, self.rewards, self.actions, self.dones = states, next_states, rewards, actions, dones

        states = torch.FloatTensor(np.asarray(self.states, dtype=np.float32))
        next_states = torch.FloatTensor(np.asarray(self.next_states, dtype=np.float32))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.dones)
        self.old_net.load_state_dict(self.net.state_dict())
        self.old_net.req_grads(False)

        if not self.fresh_advs:
            advantages, returns = self.make_advs_and_rets(states,next_states,rewards,dones,self.eval_vals)

        avg_epoch_loss, avg_epoch_policy_loss, avg_epoch_val_loss, avg_epoch_entropy = 0, 0, 0, 0

        for epoch in range(self.n_epochs):

            if self.fresh_advs:
                advantages, returns = self.make_advs_and_rets(states, next_states, rewards, dones, self.eval_vals)
            loss, epoch_loss, epoch_policy_loss, epoch_val_loss, epoch_entropy = 0,0,0,0,0
            indices = torch.randperm(states.shape[0]).long()
            self.net.train(mode=True)
            self.old_net.train(mode=True)

            for i in range(len(indices)//self.batch_size):
                self.optim.zero_grad()

                # Get data for batch
                startdx = i*self.batch_size
                endx = (i+1)*self.batch_size
                idxs = indices[startdx:endx]
                batch_data = states[idxs],actions[idxs],advantages[idxs],returns[idxs]

                # Total Loss
                policy_loss, val_loss, entropy = self.ppo_losses(*batch_data)
                loss = policy_loss + val_loss - entropy

                # Gradient Step
                loss.backward()
                self.norm = nn.utils.clip_grad_norm(self.net.parameters(), self.max_norm)
                self.net.check_grads()
                self.optim.step()
                epoch_loss += loss.data[0]
                epoch_policy_loss += policy_loss.data[0]
                epoch_val_loss += val_loss.data[0]
                epoch_entropy += entropy.data[0]
                print("Batch:", epoch*(len(indices)//self.batch_size)+i, "/", self.n_epochs*(len(indices)//self.batch_size), end="         \r")

            self.optim.zero_grad()
            avg_epoch_loss += epoch_loss/self.n_epochs
            avg_epoch_policy_loss += epoch_policy_loss/self.n_epochs
            avg_epoch_val_loss += epoch_val_loss/self.n_epochs
            avg_epoch_entropy += epoch_entropy/self.n_epochs

        self.info = {"Loss":avg_epoch_loss, "PiLoss":avg_epoch_policy_loss, "VLoss":avg_epoch_val_loss, 
                            "S":avg_epoch_entropy, "AvgDiscRew":self.avg_disc_rew, "MaxAdv":self.max_adv,
                            "MinAdv":self.min_adv, "MaxRet":self.max_ret, "MinRet":self.min_ret} 
        self.max_adv, self.min_adv, self.max_ret, self.min_ret = -1, 1, -1, 1

    def ppo_losses(self, states, actions, advs, rets):
        """
        Completes the ppo specific loss approach

        states - torch FloatTensor minibatch of states with shape (batch_size, C, H, W)
        actions - torch LongTensor minibatch of empirical actions with shape (batch_size,)
        advs - torch FloatTensor minibatch of empirical advantages with shape (batch_size,)
        rets - torch FloatTensor minibatch of empirical returns with shape (batch_size,)

        Returns:
            policy_loss - the PPO CLIP policy gradient shape (1,)
            val_loss - the critic loss shape (1,)
            entropy - the entropy of the action predictions shape (1,)
        """
        # Get new Outputs
        vals, raw_pis = self.net.forward(Variable(states))
        probs = F.softmax(raw_pis, dim=-1)
        pis = probs[torch.arange(0,len(probs)).long(), actions]

        # Get old Outputs
        old_vals, old_raw_pis = self.old_net.forward(Variable(states))
        old_vals.detach(), old_raw_pis.detach()
        old_probs = F.softmax(old_raw_pis, dim=-1)
        old_pis = old_probs[torch.arange(0,len(old_probs)).long(), actions]

        # Policy Loss
        if self.norm_batch_advs:
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)
        self.max_adv = max(torch.max(advs), self.max_adv) # Tracking variable
        self.min_adv = min(torch.min(advs), self.min_adv) # Tracking variable

        advs = Variable(advs)
        ratio = pis/(old_pis+1e-7)
        surrogate1 = ratio*advs
        surrogate2 = torch.clamp(ratio, 1.-self.epsilon, 1.+self.epsilon)*advs
        min_surr = torch.min(surrogate1, surrogate2)
        policy_loss = -min_surr.mean()

        # Value loss
        rets = Variable(rets)
        if self.clip_vals:
            clipped_vals = old_vals + torch.clamp(vals-old_vals, -self.epsilon, self.epsilon)
            v1 = .5*(vals.squeeze()-rets)**2
            v2 = .5*(clipped_vals.squeeze()-rets)**2
            val_loss = self.val_const * torch.max(v1,v2).mean()
        else:
            val_loss = self.val_const * (.5*(vals.squeeze()-rets)**2).mean()

        # Entropy Loss
        softlogs = F.log_softmax(raw_pis, dim=-1)
        entropy = -self.entropy_const * torch.mean(torch.sum(softlogs*probs, dim=-1))

        return policy_loss, val_loss, entropy

    def make_advs_and_rets(self, states, next_states, rewards, dones, eval_vals):
        """
        Creates the advantages and returns.

        states - torch FloatTensor of shape (L, C, H, W)
        next_states - torch FloatTensor of shape (L, C, H, W)
        rewards - torch FloatTensor of empirical rewards (L,)
        eval_vals - boolean denoting if the net should be in eval mode when making the value predictions

        Returns:
            advantages - torch FloatTensor of shape (L,)
            returns - torch FloatTensor of shape (L,)
        """

        self.net.train(mode=not eval_vals)
        self.net.req_grads(False)
        vals, raw_pis = self.net.forward(Variable(states))
        next_vals, _ = self.net.forward(Variable(next_states))
        del next_states
        self.net.train(mode=True)
        self.net.req_grads(True)
        advantages = self.gae(rewards.squeeze(), vals.data.squeeze(), next_vals.data.squeeze(), dones.squeeze(), self.gamma, self.lambda_)
        disc_rewards = self.discount(rewards.squeeze(), dones.squeeze(), self.gamma)
        self.avg_disc_rew = disc_rewards.mean()

        if self.use_nstep_rets: returns = advantages + vals.data.squeeze()
        else: returns = disc_rewards
        self.max_ret = max(torch.max(returns), self.max_ret)
        self.min_ret = min(torch.min(returns), self.min_ret)
        if self.norm_advs:
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-5)
        if self.norm_returns:
            returns = (returns-returns.mean())/(returns.std()+1e-5)

        return advantages, returns

    def gae(self, rewards, values, next_vals, dones, gamma, lambda_):
        """
        Performs Generalized Advantage Estimation

        rewards - torch FloatTensor of actual rewards collected. Size = L
        values - torch FloatTensor of value predictions. Size = L
        next_vals - torch FloatTensor of value predictions. Size = L
        dones - torch FloatTensor of done signals. Size = L
        gamma - float discount factor
        lambda_ - float gae moving average factor

        Returns
         advantages - torch FloatTensor of genralized advantage estimations. Size = L
        """

        deltas = rewards + gamma*next_vals*(1-dones) - values
        advantages = self.discount(deltas, dones, gamma*lambda_)

        return advantages

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
        print("\n".join([key+": "+str(round(val,5)) for key,val in sorted(self.info.items())]))

    def log_statistics(self, log, T, reward, avg_action):
        log.write("Step:"+str(T)+" – "+" – ".join([key+": "+str(round(val,5)) if "ntropy" not in key else key+": "+str(val) for key,val in self.info.items()]+["EpRew: "+str(reward), "AvgAction: "+str(avg_action)]) + '\n')
        log.flush()

    def save_model(self, net_file_name, optim_file_name):
        """
        Saves the state dict of the model to file.

        file_name - string name of the file to save the state_dict to
        """
        torch.save(self.net.state_dict(), net_file_name)
        torch.save(self.optim.state_dict(), optim_file_name)
