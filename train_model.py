import sys
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
    env_type = 'BreakoutNoFrameskip-v4'

    # Hyperparameters
    gamma = .99 # Reward discount factor
    lambda_ = .97 # GAE moving average factor
    max_tsteps = 10000000
    n_envs = 10 # Number of environments
    n_tsteps = 128 # Maximum number of steps to take in an environment for one episode
    n_rollouts = 20 # Number of times to perform rollouts before updating model
    val_const = 1 # Scales the value portion of the loss function
    entropy_const = 0.01 # Scales the entropy portion of the loss function
    max_norm = 0.5 # Scales the gradients using their norm

    lr = 1e-3 # Learning rate
    n_frame_stack = 2 # number of observations to stack for a single environment state
    n_epochs = 4
    batch_size = 256 # Batch size for PPO epochs
    epsilon = .1 # PPO clipping constant
    decay_eps = True # Decays the PPO clipping constant "epsilon" at the end of every data collection
    decay_lr = False
    cache_size = 0 # Number of samples in PPO data cache
    fresh_advs = False
    clip_vals = False
    norm_returns = False
    use_nstep_rets = True
    norm_advs = False
    norm_batch_advs = False
    use_bnorm = False
    eval_vals = True
    model_type = 'dense'
    bootstrap_next = True
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
            if "entropy_const=" in str_arg: entropy_const = float(str_arg[len("entropy_const="):])
            if "max_norm=" in str_arg: max_norm = float(str_arg[len("max_norm="):])
            if "lr=" in str_arg: lr = float(str_arg[len("lr="):])
            if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
            if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
            if "n_epochs=" in str_arg: n_epochs= int(str_arg[len('n_epochs='):])
            if "batch_size=" in str_arg: batch_size= int(str_arg[len('batch_size='):])
            if "epsilon=" in str_arg: epsilon= float(str_arg[len('epsilon='):])
            if "cache_size=" in str_arg: cache_size= int(str_arg[len('cache_size='):])
            if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
            if "n_frame_stack=" in str_arg: n_frame_stack= int(str_arg[len('n_frame_stack='):])
            if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]
            if "model_type=" in str_arg: model_type = str_arg[len('model_type='):]

            if "exp_name=" in str_arg: exp_name= str_arg[len('exp_name='):]
            elif "resume=False" in str_arg: resume = False
            elif "resume" in str_arg: resume = True
            elif "render=False" in str_arg: render = False
            elif "render" in str_arg: render = True
            elif "fresh_advs=False" in str_arg: fresh_advs = False
            elif "fresh_adv=False" in str_arg: fresh_advs = False
            elif "fresh_adv" in str_arg: fresh_advs = True
            elif "decay_eps=False" in str_arg: decay_eps = False
            elif "decay_eps" in str_arg: decay_eps = True
            elif "decay_lr=False" in str_arg: decay_lr = False
            elif "decay_lr" in str_arg: decay_lr = True
            elif "clip_vals=False" in str_arg: clip_vals = False
            elif "clip_vals" in str_arg: clip_vals = True
            elif "norm_returns=False" in str_arg: norm_returns = False
            elif "norm_returns" in str_arg: norm_returns = True
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
            elif "bootstrap_next=False" == str_arg: bootstrap_next = False
            elif "bootstrap_next=True" == str_arg: bootstrap_next = True

    hyperdict = dict()
    hyperdict["exp_name"] = exp_name
    hyperdict["env_type"] = env_type
    hyperdict["model_type"] = model_type
    hyperdict["gamma"] = gamma
    hyperdict["lambda_"] = lambda_
    hyperdict["n_rollouts"] = n_rollouts
    hyperdict["n_envs"] = n_envs
    hyperdict["n_tsteps"] = n_tsteps
    hyperdict["max_tsteps"] = max_tsteps
    hyperdict["val_const"] = val_const
    hyperdict["entropy_const"] = entropy_const
    hyperdict["max_norm"] = max_norm
    hyperdict["lr"] = lr
    hyperdict["n_frame_stack"] = n_frame_stack
    hyperdict["grid_size"] = grid_size
    hyperdict["n_foods"] = n_foods
    hyperdict["n_epochs"] = n_epochs
    hyperdict["batch_size"] = batch_size
    hyperdict["epsilon"] = epsilon
    hyperdict["cache_size"] = cache_size
    hyperdict["unit_size"] = unit_size
    hyperdict["fresh_advs"] = fresh_advs
    hyperdict["clip_vals"] = clip_vals
    hyperdict["decay_eps"] = decay_eps
    hyperdict["norm_returns"] = norm_returns
    hyperdict["use_nstep_rets"] = use_nstep_rets
    hyperdict["norm_advs"] = norm_advs
    hyperdict["norm_batch_advs"] = norm_batch_advs
    hyperdict["use_bnorm"] = use_bnorm
    hyperdict["eval_vals"] = eval_vals
    hyperdict["bootstrap_next"] = bootstrap_next
    hyperdict["decay_lr"] = decay_lr
    hyperdict["resume"] = resume
    hyperdict["render"] = render

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
        collector = Collector(reward_q, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_frame_stack=n_frame_stack, net=None, n_tsteps=n_tsteps, gamma=gamma, env_type=env_type, preprocessor=Model.preprocess, bootstrap_next=bootstrap_next)
        collectors.append(collector)

    print("Obs Shape:,",collectors[0].obs_shape)
    print("Prep Shape:,",collectors[0].prepped_shape)
    print("State Shape:,",collectors[0].state_shape)
    print("Num Samples Per Update:", n_rollouts*n_tsteps)
    print("Samples Wasted in Update:", n_rollouts*n_tsteps % batch_size)

    net = Model(collectors[0].state_shape, collectors[0].action_space, bnorm=use_bnorm)
    net = cuda_if(net)
    dummy = net.forward(Variable(cuda_if(torch.zeros(2,*collectors[0].state_shape))))
    if resume:
        net.load_state_dict(torch.load(net_save_file))
    target_net = copy.deepcopy(net)
    net.share_memory()
    target_net = cuda_if(target_net)
    data_producers = []
    for i in range(len(collectors)):
        collectors[i].net = net
        data_producer = mp.Process(target=collectors[i].produce_data, args=(data_q,))
        data_producers.append(data_producer)
        data_producer.start()

    updater = Updater(target_net, lr, entropy_const=entropy_const, value_const=val_const, gamma=gamma, lambda_=lambda_, max_norm=max_norm, batch_size=batch_size, n_epochs=n_epochs, cache_size=cache_size, epsilon=epsilon, fresh_advs=fresh_advs, clip_vals=clip_vals, norm_returns=norm_returns, norm_advs=norm_advs, norm_batch_advs=norm_batch_advs, eval_vals=eval_vals, use_nstep_rets=use_nstep_rets)
    if resume:
        updater.optim.load_state_dict(torch.load(optim_save_file))

    updater.optim.zero_grad()
    updater.net.train(mode=True)
    updater.net.req_grads(True)

    epoch = 0
    T = 0
    while T < max_tsteps:
        print("Begin Epoch", epoch, "– T =", T)
        basetime = time.time()
        epoch += 1

        # Collect data
        ep_states, ep_rewards, ep_dones, ep_actions = [], [], [], []
        ep_data = [ep_states, ep_rewards, ep_dones, ep_actions]
        for i in range(n_rollouts+2*n_envs):
            data = data_q.get()
            if i >= 2*n_envs:
                for j in range(len(ep_data)):
                    ep_data[j] += data[j]
            print("Data:", i, "/", n_rollouts+2*n_envs, end="      \r")
        T += len(ep_data[0])
        if decay_eps:
            updater.epsilon = (1-T/max_tsteps)*epsilon
        if decay_lr:
            new_lr = (1-T/max_tsteps)*lr
            state_dict = updater.optim.state_dict()
            updater.optim = optim.Adam(updater.net.parameters(), lr=new_lr)
            updater.optim.load_state_dict(state_dict)

        # Calculate the Loss and Update nets
        updater.update_model(*ep_data)
        updater.save_model(net_save_file, optim_save_file)
        net.load_state_dict(updater.net.state_dict()) # update all collector nets

        # Print Epoch Data
        updater.print_statistics()
        avg_action = np.mean(ep_data[3])
        print("Grad Norm:", updater.norm, "– Avg Action:", avg_action)
        avg_reward = reward_q.get()
        reward_q.put(avg_reward)
        print("Average Reward:", avg_reward, end='\n\n')
        updater.log_statistics(log, T, avg_reward, avg_action)

        # Check for memory leaks
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Time:", time.time()-basetime)
        print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))

    # Close processes
    for dp in data_producers:
        dp.terminate()
