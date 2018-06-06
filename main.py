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
import dense_model
import conv_model
import a3c_model

def cuda_if(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj

if __name__ == '__main__':
    mp.set_start_method('forkserver')

    exp_name = 'default'
    env_type = 'Pong-v0'

    # Hyperparameters
    gamma = .99 # Reward discount factor
    lambda_ = .95 # GAE moving average factor
    max_tsteps = 10000000
    n_envs = 10 # Number of environments
    n_tsteps = 64 # Maximum number of steps to take in an environment for one episode
    n_rollouts = 32 # Number of times to perform rollouts before updating model
    val_const = .5 # Scales the value portion of the loss function
    entr_coef = .01 # Scales the value portion of the loss function
    entr_coef_low = .001 # Lower bound for entropy coefficient decay
    max_norm = 0.5 # Scales the gradients using their norm
    lr = 1e-4 # Learning rate
    lr_low = 1e-5 # Learning rate lower bound
    n_frame_stack = 4 # number of observations to stack for a single environment state
    n_epochs = 4
    batch_size = 256 # Batch size for PPO epochs
    epsilon = .15 # PPO clipping constant
    epsilon_low = .05 # PPO clipping constant lower bound
    cache_size = 0 # Number of samples in PPO data cache
    n_past_rews = 25 # Used to track reward spread

    collector_cuda = True # Determines if collector networks are loaded to GPU
    decay_eps = True # Decays the PPO clipping constant "epsilon" at the end of every data collection
    decay_lr = True
    decay_entr = True
    clip_vals = False
    use_nstep_rets = True
    norm_advs = True
    norm_batch_advs = False
    use_bnorm = False
    eval_vals = True
    model_type = 'conv' # Options include: 'conv', 'dense', 'a3c'
    optim_type = 'rmsprop' # Options include: 'adam', 'rmsprop'
    resume = False
    render = False

    # Environment Choices
    grid_size = [15,15]
    n_foods = 2
    unit_size = 4

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            str_arg = str(arg)
            if "gamma=" in str_arg: gamma = float(str_arg[len("gamma="):])
            if "lambda_=" in str_arg: lambda_ = float(str_arg[len("lambda_="):])
            if "n_rollouts=" in str_arg: n_rollouts = int(str_arg[len("n_rollouts="):])
            if "n_envs=" in str_arg: n_envs = int(str_arg[len("n_envs="):])
            if "n_tsteps=" in str_arg: n_tsteps = int(str_arg[len("n_tsteps="):])
            if "max_tsteps=" in str_arg: max_tsteps = int(str_arg[len("max_tsteps="):])
            if "val_const=" in str_arg: val_const = float(str_arg[len("val_const="):])
            if "entr_coef=" in str_arg: entr_coef = float(str_arg[len("entr_coef="):])
            if "entr_coef_high=" in str_arg: entr_coef = float(str_arg[len("entr_coef_high="):])
            if "entr_coef_low=" in str_arg: entr_coef_low = float(str_arg[len("entr_coef_low="):])
            if "max_norm=" in str_arg: max_norm = float(str_arg[len("max_norm="):])
            if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
            if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
            if "n_epochs=" in str_arg: n_epochs= int(str_arg[len('n_epochs='):])
            if "batch_size=" in str_arg: batch_size= int(str_arg[len('batch_size='):])
            if "epsilon=" in str_arg: epsilon= float(str_arg[len('epsilon='):])
            if "epsilon_low=" in str_arg: epsilon_low= float(str_arg[len('epsilon_low='):])
            if "cache_size=" in str_arg: cache_size= int(str_arg[len('cache_size='):])
            if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
            if "n_frame_stack=" in str_arg: n_frame_stack= int(str_arg[len('n_frame_stack='):])
            if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]
            if "model_type=" in str_arg: model_type = str_arg[len('model_type='):]
            if "optim_type=" in str_arg: optim_type = str_arg[len('optim_type='):]

            if "exp_name=" in str_arg: exp_name= str_arg[len('exp_name='):]
            elif "resume=False" in str_arg: resume = False
            elif "resume" in str_arg: resume = True
            elif "render=False" in str_arg: render = False
            elif "render" in str_arg: render = True
            elif "collector_cuda=False" == str_arg: collector_cuda = False
            elif "collector_cuda=True" == str_arg: collector_cuda = True
            elif "decay_eps=False" in str_arg: decay_eps = False
            elif "decay_eps" in str_arg: decay_eps = True
            elif "decay_entr=False" in str_arg: decay_entr = False
            elif "decay_entr=True" == str_arg: decay_entr = True
            elif "decay_lr=False" in str_arg: decay_lr = False
            elif "decay_lr" in str_arg: decay_lr = True
            elif "lr=" in str_arg: lr = float(str_arg[len("lr="):])
            elif "lr_low=" in str_arg: lr_low = float(str_arg[len("lr_low="):])
            elif "clip_vals=False" in str_arg: clip_vals = False
            elif "clip_vals" in str_arg: clip_vals = True
            elif "use_nstep_rets=False" in str_arg: use_nstep_rets = False
            elif "use_nstep_rets" in str_arg: use_nstep_rets = True
            elif "norm_advs=False" in str_arg: norm_advs = False
            elif "norm_advs" in str_arg: norm_advs = True
            elif "norm_batch_advs=False" in str_arg: norm_batch_advs = False
            elif "norm_batch_advs" in str_arg: norm_batch_advs = True
            elif "use_bnorm=False" in str_arg: use_bnorm = False
            elif "use_bnorm" in str_arg: use_bnorm = True
            elif "eval_vals=False" in str_arg: eval_vals = False
            elif "eval_vals" in str_arg: eval_vals = True

    hyperdict = dict()
    hyperdict["exp_name"] = exp_name
    hyperdict["env_type"] = env_type
    hyperdict["model_type"] = model_type
    hyperdict["optim_type"] = optim_type
    hyperdict["gamma"] = gamma
    hyperdict["lambda_"] = lambda_
    hyperdict["n_rollouts"] = n_rollouts
    hyperdict["n_envs"] = n_envs
    hyperdict["n_tsteps"] = n_tsteps
    hyperdict["max_tsteps"] = max_tsteps
    hyperdict["val_const"] = val_const
    hyperdict["entr_coef"] = entr_coef
    hyperdict["entr_coef_low"] = entr_coef_low
    hyperdict["max_norm"] = max_norm
    hyperdict["lr"] = lr
    hyperdict["lr_low"] = lr_low
    hyperdict["epsilon_low"] = epsilon_low
    hyperdict["n_frame_stack"] = n_frame_stack
    hyperdict["grid_size"] = grid_size
    hyperdict["n_foods"] = n_foods
    hyperdict["n_epochs"] = n_epochs
    hyperdict["batch_size"] = batch_size
    hyperdict["epsilon"] = epsilon
    hyperdict["cache_size"] = cache_size
    hyperdict["unit_size"] = unit_size
    hyperdict["clip_vals"] = clip_vals
    hyperdict["decay_eps"] = decay_eps
    hyperdict["collector_cuda"] = collector_cuda
    hyperdict["use_nstep_rets"] = use_nstep_rets
    hyperdict["norm_advs"] = norm_advs
    hyperdict["norm_batch_advs"] = norm_batch_advs
    hyperdict["use_bnorm"] = use_bnorm
    hyperdict["eval_vals"] = eval_vals
    hyperdict["decay_lr"] = decay_lr
    hyperdict["decay_entr"] = decay_entr
    hyperdict["resume"] = resume
    hyperdict["render"] = render

    entr_coef_diff = entr_coef - entr_coef_low
    epsilon_diff = epsilon - epsilon_low
    lr_diff = lr - lr_low

    if batch_size > n_rollouts*n_tsteps:
        batch_size = n_rollouts*n_tsteps

    items = list(hyperdict.items())
    for k, v in sorted(items):
        print(k+":", v)

    if 'dense' in model_type:
        Model = dense_model.Model
    elif 'conv' in model_type:
        Model = conv_model.Model
    elif 'a3c' in model_type:
        Model = a3c_model.Model
    else:
        Model = dense_model.Model
    net_save_file = exp_name+"_net.p"
    best_net_file = exp_name+"_best.p"
    optim_save_file = exp_name+"_optim.p"
    log_file = exp_name+"_log.txt"
    if resume: log = open(log_file, 'a')
    else: log = open(log_file, 'w')
    for k, v in sorted(items):
        log.write(k+":"+str(v)+"\n")

    # Shared Data Objects
    data_q = mp.Queue(n_envs)
    reward_q = mp.Queue(1)
    reward_q.put(-1)

    collectors = []
    for i in range(n_envs):
        collector = Collector(reward_q, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_frame_stack=n_frame_stack, net=None, n_tsteps=n_tsteps, gamma=gamma, env_type=env_type, preprocessor=Model.preprocess, use_cuda=collector_cuda)
        collectors.append(collector)

    print("Obs Shape:,",collectors[0].obs_shape)
    print("Prep Shape:,",collectors[0].prepped_shape)
    print("State Shape:,",collectors[0].state_shape)
    print("Num Samples Per Update:", n_rollouts*n_tsteps)
    print("Samples Wasted in Update:", n_rollouts*n_tsteps % batch_size)

    net = Model(collectors[0].state_shape, collectors[0].action_space, bnorm=use_bnorm)
    dummy = net.forward(Variable(torch.zeros(2,*collectors[0].state_shape)))
    if resume:
        net.load_state_dict(torch.load(net_save_file))
    target_net = copy.deepcopy(net)
    net.share_memory()
    target_net = cuda_if(target_net)
    data_producers = []
    for i in range(len(collectors)):
        collectors[i].load_net(net)
        data_producer = mp.Process(target=collectors[i].produce_data, args=(data_q,))
        data_producers.append(data_producer)
        data_producer.start()

    updater = Updater(target_net, lr, entr_coef=entr_coef, value_const=val_const, gamma=gamma, lambda_=lambda_, max_norm=max_norm, batch_size=batch_size, n_epochs=n_epochs, cache_size=cache_size, epsilon=epsilon, clip_vals=clip_vals, norm_advs=norm_advs, norm_batch_advs=norm_batch_advs, eval_vals=eval_vals, use_nstep_rets=use_nstep_rets, optim_type=optim_type)
    if resume:
        updater.optim.load_state_dict(torch.load(optim_save_file))

    updater.optim.zero_grad()
    updater.net.train(mode=True)
    updater.net.req_grads(True)

    logger = Logger()
    past_rews = [-1]*n_past_rews
    best_avg_rew = 0
    epoch = 0
    T = 0
    while T < max_tsteps:
        print("Begin Epoch", epoch, "– T =", T)
        basetime = time.time()
        epoch += 1

        # Collect data
        ep_states, ep_nexts, ep_rewards, ep_dones, ep_actions = [], [], [], [], []
        ep_data = [ep_states, ep_nexts, ep_rewards, ep_dones, ep_actions]
        num_rolls = 2*n_rollouts
        for i in range(num_rolls):
            data = data_q.get()
            if i >= n_rollouts:
                for j in range(len(ep_data)):
                    ep_data[j] += data[j]
            print("Data:", i, "/", num_rolls, end="      \r")
        T += len(ep_data[0])
        if decay_eps:
            updater.epsilon = (1-T/(max_tsteps))*epsilon_diff + epsilon_low
        if decay_lr:
            new_lr = (1-T/(max_tsteps))*lr_diff + lr_low
            updater.new_lr(new_lr)
        if decay_entr:
            updater.entr_coef = entr_coef_diff*(1-T/(max_tsteps))+entr_coef_low

        # Reward Stats
        avg_reward = reward_q.get()
        reward_q.put(avg_reward)
        past_rews.append(avg_reward)
        past_rews = past_rews[-n_past_rews:]
        past_rews_sort = sorted(past_rews)
        if avg_reward > best_avg_rew:
            best_avg_rew = avg_reward
            updater.save_model(best_net_file, None)

        # Calculate the Loss and Update nets
        updater.update_model(*ep_data)
        updater.save_model(net_save_file, optim_save_file)
        net.load_state_dict(updater.net.state_dict()) # update all collector nets

        # Print Epoch Data
        updater.print_statistics()
        avg_action = np.mean(ep_data[4])
        print("Grad Norm:", float(updater.norm), "– Avg Action:", avg_action, "– Best AvgRew:", best_avg_rew)
        print("Avg Rew:", avg_reward, "– High:", past_rews_sort[-1], "– Low:", past_rews_sort[0], end='\n')
        updater.log_statistics(log, T, avg_reward, avg_action)
        updater.info['AvgRew'] = avg_reward
        logger.append(updater.info, x_val=T)

        # Check for memory leaks
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Time:", time.time()-basetime)
        print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))

    logger.make_plots(exp_name)
    # Close processes
    for dp in data_producers:
        dp.terminate()
