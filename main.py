import sys
from logger import Logger
from collector import Collector
from updater import Updater
import torch
from torch.autograd import Variable
import numpy as np
import gc
import resource
import torch.multiprocessing as mp
import copy
import time
from hyperparameters import HyperParams
import dense_model
import conv_model
import a3c_model
import queue

def cuda_if(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj

if __name__ == '__main__':
    mp.set_start_method('forkserver')

    # Set Hyperparameters
    hyper_params = HyperParams()
    hyps = hyper_params.hyps

    # Print Hyperparameters To Screen
    items = list(hyps.items())
    for k, v in sorted(items):
        print(k+":", v)

    # Set Variables
    if 'dense' in hyps['model_type']:
        Model = dense_model.Model
    elif 'conv' in hyps['model_type']:
        Model = conv_model.Model
    elif 'a3c' in hyps['model_type']:
        Model = a3c_model.Model
    else:
        Model = dense_model.Model
    net_save_file = hyps['exp_name']+"_net.p"
    best_net_file = hyps['exp_name']+"_best.p"
    best_by_diff_file = hyps['exp_name']+"_bestdiff.p"
    optim_save_file = hyps['exp_name']+"_optim.p"
    log_file = hyps['exp_name']+"_log.txt"
    if hyps['resume']: log = open(log_file, 'a')
    else: log = open(log_file, 'w')
    for k, v in sorted(items):
        log.write(k+":"+str(v)+"\n")

    # Make Collectors
    data_q = mp.Queue(hyps['n_envs'])
    reward_q = mp.Queue(1)
    reward_q.put(-1)
    gate_qs = []
    collectors = []
    for i in range(hyps['n_envs']):
        gate_q = mp.Queue(3) # Used to control rollout collections
        collector = Collector(gate_q, reward_q, grid_size=hyps['grid_size'], n_foods=hyps['n_foods'], unit_size=hyps['unit_size'], n_frame_stack=hyps['n_frame_stack'], net=None, n_tsteps=hyps['n_tsteps'], gamma=hyps['gamma'], env_type=hyps['env_type'], preprocessor=Model.preprocess, use_cuda=hyps['collector_cuda'])
        collectors.append(collector)
        gate_qs.append(gate_q)
    for g in range(hyps['n_rollouts']):
        gate_qs[g%len(gate_qs)].put(True)
    print("Obs Shape:,",collectors[0].obs_shape)
    print("Prep Shape:,",collectors[0].prepped_shape)
    print("State Shape:,",collectors[0].state_shape)
    print("Num Samples Per Update:", hyps['n_rollouts']*hyps['n_tsteps'])
    print("Samples Wasted in Update:", hyps['n_rollouts']*hyps['n_tsteps'] % hyps['batch_size'])

    # Make Network
    net = Model(collectors[0].state_shape, collectors[0].action_space, bnorm=hyps['use_bnorm'])
    dummy = net.forward(Variable(torch.zeros(2,*collectors[0].state_shape)))
    if hyps['resume']:
        net.load_state_dict(torch.load(net_save_file))
    target_net = copy.deepcopy(net)
    net.share_memory()
    target_net = cuda_if(target_net)

    # Start Data Collection
    print("Making New Processes")
    data_producers = []
    for i in range(len(collectors)):
        collectors[i].load_net(net)
        data_producer = mp.Process(target=collectors[i].produce_data, args=(data_q,))
        data_producers.append(data_producer)
        data_producer.start()
        print(i, "/", len(collectors), end='\r')

    # Make Updater
    updater = Updater(target_net, hyps['lr'], entr_coef=hyps['entr_coef'], value_const=hyps['val_const'], gamma=hyps['gamma'], lambda_=hyps['lambda_'], max_norm=hyps['max_norm'], batch_size=hyps['batch_size'], n_epochs=hyps['n_epochs'], cache_size=hyps['cache_size'], epsilon=hyps['epsilon'], clip_vals=hyps['clip_vals'], norm_advs=hyps['norm_advs'], norm_batch_advs=hyps['norm_batch_advs'], eval_vals=hyps['eval_vals'], use_nstep_rets=hyps['use_nstep_rets'], optim_type=hyps['optim_type'])
    if hyps['resume']:
        updater.optim.load_state_dict(torch.load(optim_save_file))
    updater.optim.zero_grad()
    updater.net.train(mode=True)
    updater.net.req_grads(True)

    # Decay Precursors
    entr_coef_diff = hyps['entr_coef'] - hyps['entr_coef_low']
    epsilon_diff = hyps['epsilon'] - hyps['epsilon_low']
    lr_diff = hyps['lr'] - hyps['lr_low']

    logger = Logger()
    idx_perm = np.random.permutation(len(gate_qs)).astype(np.int)
    past_rews = queue.Queue(hyps['n_past_rews'])
    for i in range(hyps['n_past_rews']):
        past_rews.put(0)
    last_avg_rew = 0
    best_rew_diff = 0
    best_avg_rew = 0
    epoch = 0
    T = 0
    while T < hyps['max_tsteps']:
        print("Begin Epoch", epoch, "– T =", T)
        basetime = time.time()
        epoch += 1

        # Collect data
        ep_states, ep_nexts, ep_rewards, ep_dones, ep_actions = [], [], [], [], []
        ep_data = [ep_states, ep_nexts, ep_rewards, ep_dones, ep_actions]
        for i in range(hyps['n_rollouts']):
            data = data_q.get()
            for j in range(len(ep_data)):
                ep_data[j] += data[j]
            print("Data:", i, "/", hyps['n_rollouts'], end="      \r")
        T += len(ep_data[0])
        if hyps['decay_eps']:
            updater.epsilon = (1-T/(hyps['max_tsteps']))*epsilon_diff + hyps['epsilon_low']
        if hyps['decay_lr']:
            new_lr = (1-T/(hyps['max_tsteps']))*lr_diff + hyps['lr_low']
            updater.new_lr(new_lr)
        if hyps['decay_entr']:
            updater.entr_coef = entr_coef_diff*(1-T/(hyps['max_tsteps']))+hyps['entr_coef_low']

        # Calculate the Loss and Update nets
        updater.update_model(*ep_data)
        net.load_state_dict(updater.net.state_dict()) # update all collector nets
        
        # Resume Data Collection
        for g in range(hyps['n_rollouts']):
            idx = idx_perm[g%len(gate_qs)]
            gate_qs[idx].put(True)
        idx_perm = np.random.permutation(len(gate_qs)).astype(np.int)

        # Reward Stats
        avg_reward = reward_q.get()
        reward_q.put(avg_reward)
        past_rews.get()
        past_rews.put(avg_reward)
        past_rews_sort = sorted(past_rews)
        avg_rew_diff = avg_reward - last_avg_rew
        last_avg_rew = avg_reward

        # Save Model
        if avg_rew_diff > best_rew_diff:
            best_rew_diff = avg_rew_diff
            updater.save_model(best_by_diff_file, None)
        if avg_reward > best_avg_rew:
            best_avg_rew = avg_reward
            updater.save_model(best_net_file, None)
        if epoch % 10 == 0:
            updater.save_model(net_save_file, optim_save_file)

        # Print Epoch Data
        updater.print_statistics()
        avg_action = np.mean(ep_data[4])
        print("Grad Norm:", float(updater.norm), "– Avg Action:", avg_action, "– Best AvgRew:", best_avg_rew, "– Best Diff:", best_rew_diff)
        print("Avg Rew:", avg_reward, "– High:", past_rews_sort[-1], "– Low:", past_rews_sort[0], end='\n')
        updater.log_statistics(log, T, avg_reward, avg_action)
        updater.info['AvgRew'] = avg_reward
        logger.append(updater.info, x_val=T)

        # Check for memory leaks
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Time:", time.time()-basetime)
        print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))

    logger.make_plots(hyps['exp_name'])
    # Close processes
    for dp in data_producers:
        dp.terminate()
