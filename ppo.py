import os
import sys
import gym
from logger import Logger
from runner import Runner
from updater import Updater
import torch
from torch.autograd import Variable
import numpy as np
import gc
import resource
import torch.multiprocessing as mp
import copy
import time
from collections import deque
from utils import cuda_if, deque_maxmin

class PPO:
    def __init__(self):
        pass

    def train(self, hyps): 
        """
        hyps - dictionary of required hyperparameters
            type: dict
        """

        # Print Hyperparameters To Screen
        items = list(hyps.items())
        for k, v in sorted(items):
            print(k+":", v)

        # Make Save Files
        if "save_folder" in hyps:
            save_folder = hyps['save_folder']
        else:
            save_folder = "./saved_data/"

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        base_name = save_folder + hyps['exp_name']
        net_save_file = base_name+"_net.p"
        best_net_file = base_name+"_best.p"
        optim_save_file = base_name+"_optim.p"
        log_file = base_name+"_log.txt"
        if hyps['resume']: log = open(log_file, 'a')
        else: log = open(log_file, 'w')
        for k, v in sorted(items):
            log.write(k+":"+str(v)+"\n")

        # Miscellaneous Variable Prep
        logger = Logger()
        shared_len = hyps['n_tsteps']*hyps['n_rollouts']
        env = gym.make(hyps['env_type'])
        obs = env.reset()
        prepped = hyps['preprocess'](obs)
        hyps['state_shape'] = [hyps['n_frame_stack']] + [*prepped.shape[1:]]
        if hyps['env_type'] == "Pong-v0":
            action_size = 3
        else:
            action_size = env.action_space.n*(hyps['env_type']!="Pong-v0")
        hyps['action_shift'] = (4-action_size)*(hyps['env_type']=="Pong-v0") 
        print("Obs Shape:,",obs.shape)
        print("Prep Shape:,",prepped.shape)
        print("State Shape:,",hyps['state_shape'])
        print("Num Samples Per Update:", shared_len)
        print("Samples Wasted in Update:", shared_len % hyps['batch_size'])
        del env

        # Prepare Shared Variables
        shared_data = {'states': cuda_if(torch.zeros(shared_len, *hyps['state_shape']).share_memory_()),
                'next_states': cuda_if(torch.zeros(shared_len, *hyps['state_shape']).share_memory_()),
                'rewards': cuda_if(torch.zeros(shared_len).share_memory_()),
                'actions': torch.zeros(shared_len).long().share_memory_(),
                'dones': cuda_if(torch.zeros(shared_len).share_memory_())}
        n_rollouts = hyps['n_rollouts']
        gate_q = mp.Queue(n_rollouts)
        stop_q = mp.Queue(n_rollouts)
        reward_q = mp.Queue(1)
        reward_q.put(-1)

        # Make Runners
        runners = []
        for i in range(hyps['n_envs']):
            runner = Runner(shared_data, hyps, gate_q, stop_q, reward_q)
            runners.append(runner)

        # Make Network
        net = hyps['model'](hyps['state_shape'], action_size, bnorm=hyps['use_bnorm'])
        if hyps['resume']:
            net.load_state_dict(torch.load(net_save_file))
        base_net = copy.deepcopy(net)
        net = cuda_if(net)
        net.share_memory()
        base_net = cuda_if(base_net)

        # Start Data Collection
        print("Making New Processes")
        procs = []
        for i in range(len(runners)):
            proc = mp.Process(target=runners[i].run, args=(net,))
            procs.append(proc)
            proc.start()
            print(i, "/", len(runners), end='\r')
        for i in range(n_rollouts):
            gate_q.put(i)

        # Make Updater
        updater = Updater(base_net, hyps['lr'], entr_coef=hyps['entr_coef'], value_const=hyps['val_const'], gamma=hyps['gamma'], lambda_=hyps['lambda_'], max_norm=hyps['max_norm'], batch_size=hyps['batch_size'], n_epochs=hyps['n_epochs'], epsilon=hyps['epsilon'], clip_vals=hyps['clip_vals'], norm_advs=hyps['norm_advs'], norm_batch_advs=hyps['norm_batch_advs'], use_nstep_rets=hyps['use_nstep_rets'], optim_type=hyps['optim_type'])
        if hyps['resume']:
            updater.optim.load_state_dict(torch.load(optim_save_file))
        updater.optim.zero_grad()
        updater.net.train(mode=True)
        updater.net.req_grads(True)

        # Prepare Decay Precursors
        entr_coef_diff = hyps['entr_coef'] - hyps['entr_coef_low']
        epsilon_diff = hyps['epsilon'] - hyps['epsilon_low']
        lr_diff = hyps['lr'] - hyps['lr_low']
        gamma_diff = hyps['gamma_high'] - hyps['gamma']

        # Training Loop
        past_rews = deque([0]*hyps['n_past_rews'])
        last_avg_rew = 0
        best_rew_diff = 0
        best_avg_rew = 0
        epoch = 0
        T = 0
        while T < hyps['max_tsteps']:
            basetime = time.time()
            epoch += 1

            # Collect data
            for i in range(n_rollouts):
                stop_q.get()
            T += shared_len

            # Reward Stats
            avg_reward = reward_q.get()
            reward_q.put(avg_reward)
            last_avg_rew = avg_reward
            if avg_reward > best_avg_rew:
                best_avg_rew = avg_reward
                updater.save_model(best_net_file, None)

            # Calculate the Loss and Update nets
            updater.update_model(shared_data)
            net.load_state_dict(updater.net.state_dict()) # update all collector nets
            
            # Resume Data Collection
            for i in range(n_rollouts):
                gate_q.put(i)

            # Decay HyperParameters
            if hyps['decay_eps']:
                updater.epsilon = (1-T/(hyps['max_tsteps']))*epsilon_diff + hyps['epsilon_low']
                print("New Eps:", updater.epsilon)
            if hyps['decay_lr']:
                new_lr = (1-T/(hyps['max_tsteps']))*lr_diff + hyps['lr_low']
                updater.new_lr(new_lr)
                print("New lr:", new_lr)
            if hyps['decay_entr']:
                updater.entr_coef = entr_coef_diff*(1-T/(hyps['max_tsteps']))+hyps['entr_coef_low']
                print("New Entr:", updater.entr_coef)
            if hyps['incr_gamma']:
                updater.gamma = gamma_diff*(T/(hyps['max_tsteps']))+hyps['gamma']
                print("New Gamma:", updater.gamma)

            # Periodically save model
            if epoch % 10 == 0:
                updater.save_model(net_save_file, optim_save_file)

            # Print Epoch Data
            past_rews.popleft()
            past_rews.append(avg_reward)
            max_rew, min_rew = deque_maxmin(past_rews)
            updater.print_statistics()
            avg_action = shared_data['actions'].float().mean().item()
            print("Epoch", epoch, "– T =", T)
            print("Grad Norm:",float(updater.norm),"– Avg Action:",avg_action,"– Best AvgRew:",best_avg_rew)
            print("Avg Rew:", avg_reward, "– High:", max_rew, "– Low:", min_rew, end='\n')
            updater.log_statistics(log, T, avg_reward, avg_action, best_avg_rew)
            updater.info['AvgRew'] = avg_reward
            logger.append(updater.info, x_val=T)

            # Check for memory leaks
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print("Time:", time.time()-basetime)
            print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))

        logger.make_plots(hyps['exp_name'])
        # Close processes
        for p in procs:
            p.terminate()
