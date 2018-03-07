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

    def __init__(self, net, lr, entropy_const=0.01, value_const=0.5, gamma=0.99, lambda_=0.98, max_norm=0.5, n_epochs=5, batch_size=200, cache_size=3000, epsilon=.2, fresh_advs=False, clip_vals=False, norm_returns=False, norm_advs=True, norm_batch_advs=False):
        self.net = net
        self.old_net = copy.deepcopy(self.net)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.global_loss = 0 # Used for efficiency in backprop
        self.entropy_const = entropy_const
        self.val_const = value_const
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_norm = max_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.fresh_advs = fresh_advs
        self.clip_vals = clip_vals
        self.norm_returns = norm_returns
        self.norm_advs = norm_advs
        self.norm_batch_advs = norm_batch_advs

        # Data caches
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.advantages = []

        # Tracking variables
        self.avg_loss = None
        self.info = {}

    def update_model(self, states, rewards, dones, actions):
        """
        This function accepts the data collected from a rollout and performs PPO update iterations
        on the neural net.

        states - python list of the environment states from the rollouts
                shape = (n_states, *state_shape)
        rewards - python list of rewards from the rollouts
                shape = (n_states,)
        dones - python list of done signals from the rollouts
                dones = (n_states,)
        actions - python integer list denoting the actual action
                indexes taken in the rollout
        """

        if self.cache_size > 0:
            self.states += states
            self.states = self.states[-self.cache_size:]
            self.rewards += rewards
            self.rewards = self.rewards[-self.cache_size:]
            self.actions += actions
            self.actions = self.actions[-self.cache_size:]
            self.dones += dones
            self.dones = self.dones[-self.cache_size:]
        else:
            self.states = states
            self.rewards = rewards
            self.actions = actions
            self.dones = dones

        states = torch.FloatTensor(np.asarray(self.states, dtype=np.float32))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.dones)
        self.old_net.load_state_dict(self.net.state_dict())
        self.old_net.req_grads(False)

        if not self.fresh_advs:
            self.net.train(mode=False)
            vals, raw_pis = self.net.forward(Variable(states))
            self.net.train(mode=True)
            gae_vals = torch.cat([vals.data.squeeze(), cuda_if(torch.zeros(1))], 0)
            advantages = self.gae(rewards.squeeze(), gae_vals.squeeze(), dones.squeeze(), self.gamma, self.lambda_)
            returns = advantages + vals.data.squeeze()
            if self.norm_advs:
                advantages = (advantages-advantages.mean())/(advantages.std()+1e-5)
            if self.norm_returns:
                returns = (returns-returns.mean())/(returns.std()+1e-5)

        total_epoch_loss = 0
        total_epoch_policy_loss = 0
        total_epoch_val_loss = 0
        total_epoch_entropy = 0

        for epoch in range(self.n_epochs):

            if self.fresh_advs:
                self.net.train(mode=False)
                vals, raw_pis = self.net.forward(Variable(states))
                self.net.train(mode=True)
                gae_vals = torch.cat([vals.data.squeeze(), cuda_if(torch.zeros(1))], 0)
                advantages = self.gae(rewards.squeeze(), gae_vals.squeeze(), dones.squeeze(), self.gamma, self.lambda_)
                returns = advantages + vals.data.squeeze()
                if self.norm_advs:
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-5)
                if self.norm_returns:
                    returns = (returns-returns.mean())/(returns.std()+1e-5)

            loss = 0
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_val_loss = 0
            epoch_entropy = 0

            indices = torch.randperm(states.shape[0]).long()

            for i in range(0,indices.shape[0]-self.batch_size, self.batch_size):
                self.optim.zero_grad()

                # Get data for batch
                batch_idxs = indices[i:i+self.batch_size]
                batch_states, batch_returns = Variable(states[batch_idxs]), Variable(returns[batch_idxs])
                batch_actions, batch_advs = actions[batch_idxs], advantages[batch_idxs]

                # Get new Outputs
                vals, raw_pis = self.net.forward(batch_states)
                probs = F.softmax(raw_pis, dim=-1)
                pis = probs[torch.arange(0,probs.shape[0]).long(), batch_actions]

                # Get old Outputs
                old_vals, old_raw_pis = self.old_net.forward(batch_states)
                old_vals.detach(), old_raw_pis.detach()
                old_probs = F.softmax(old_raw_pis, dim=-1)
                old_pis = old_probs[torch.arange(0,old_probs.shape[0]).long(), batch_actions]

                # Policy Loss
                if self.norm_batch_advs:
                    batch_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-5)
                batch_advs = Variable(batch_advs)
                ratio = pis/(old_pis+1e-10) * batch_advs
                clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*batch_advs
                policy_loss = -torch.min(ratio, clipped_ratio).mean()

                # Value loss
                if self.clip_vals:
                    clipped_vals = old_vals + torch.clamp(vals-old_vals, -self.epsilon, self.epsilon)
                    v1 = .5*(vals.squeeze()-batch_returns)**2
                    v2 = .5*(clipped_vals.squeeze()-batch_returns)**2
                    val_loss = self.val_const * torch.max(v1,v2).mean()
                else:
                    val_loss = self.val_const * (.5*(vals.squeeze()-batch_returns)**2).mean()

                # Entropy Loss
                softlogs = F.log_softmax(raw_pis, dim=-1)
                entropy = -self.entropy_const * torch.mean(torch.sum(softlogs*probs, dim=-1))

                # Total Loss
                loss = policy_loss + val_loss - entropy

                # Gradient Step
                loss.backward()
                self.norm = nn.utils.clip_grad_norm(self.net.parameters(), self.max_norm)
                self.optim.step()
                epoch_loss += loss.data[0]
                epoch_policy_loss += policy_loss.data[0]
                epoch_val_loss += val_loss.data[0]
                epoch_entropy += entropy.data[0]

            self.optim.zero_grad()
            total_epoch_loss += epoch_loss
            total_epoch_policy_loss += epoch_policy_loss
            total_epoch_val_loss += epoch_val_loss
            total_epoch_entropy += epoch_entropy
            if self.avg_loss is None:
                self.avg_loss = epoch_loss
                self.avg_policy_loss = epoch_policy_loss
                self.avg_val_loss = epoch_val_loss
                self.avg_entropy = epoch_entropy
            else:
                self.avg_loss = .99*self.avg_loss + .01*epoch_loss
                self.avg_policy_loss = .99*self.avg_policy_loss + .01*epoch_policy_loss
                self.avg_val_loss = .99*self.avg_val_loss + .01*epoch_val_loss
                self.avg_entropy = .99*self.avg_entropy + .01*epoch_entropy

            self.info = {"Global Loss":self.avg_loss,
                        "Policy Loss":self.avg_policy_loss,
                        "Value Loss":self.avg_val_loss,
                        "Entropy":self.avg_entropy}
        print("Update Avgs – L:", total_epoch_loss/self.n_epochs, "– PL:", total_epoch_policy_loss/self.n_epochs, "– VL:", total_epoch_val_loss/self.n_epochs, "– S:", total_epoch_entropy/self.n_epochs)

    def gae(self, rewards, values, dones, gamma, lambda_):
        """
        Performs Generalized Advantage Estimation

        rewards - torch FloatTensor of actual rewards collected. Size = L
        values - torch FloatTensor of value predictions. Size = L+1
        dones - torch FloatTensor of done signals. Size = L
        gamma - float discount factor
        lambda_ - float gae moving average factor

        returns - torch FloatTensor of genralized advantage estimations. Size = L
        """

        deltas = rewards + gamma*values[1:]*(1-dones) - values[:-1]
        advantages = self.discount(deltas, dones, gamma*lambda_)

        #test_deltas = torch.zeros(len(rewards))
        #test_advantages = torch.zeros(len(rewards))
        #running_adv = 0
        #for i in reversed(range(len(rewards))):
        #    delta = rewards[i] + gamma*values[i+1]*(1-dones[i]) - values[i]
        #    test_deltas[i] = delta
        #    test_advantages[i] = running_adv = delta + gamma*lambda_*running_adv*(1-dones[i])
        #print("Delta", np.array_equal(deltas.numpy(), test_deltas.numpy()))
        #print("Deltadist", np.sqrt(torch.sum((deltas - test_deltas)**2)))
        #print("Advantages", np.array_equal(advantages.numpy(), test_advantages.numpy()))
        #print("Advadist", np.sqrt(torch.sum((advantages - test_advantages)**2)))

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
        print("Running Avgs:"+" – ".join([key+": "+str(round(val,5)) if "ntropy" not in key else key+": "+str(val) for key,val in self.info.items()]))

    def log_statistics(self, log, T, reward):
        log.write("Step:"+str(T)+" – "+" – ".join([key+": "+str(round(val,5)) if "ntropy" not in key else key+": "+str(val) for key,val in self.info.items()]+["EpRew: "+str(reward)]) + '\n')
        log.flush()

    def save_model(self, net_file_name, optim_file_name):
        """
        Saves the state dict of the model to file.

        file_name - string name of the file to save the state_dict to
        """
        torch.save(self.net.state_dict(), net_file_name)
        torch.save(self.optim.state_dict(), optim_file_name)
