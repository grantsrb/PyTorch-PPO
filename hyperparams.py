import sys
import preprocessing
from models import ConvModel, FCModel, A3CModel, GRUModel
import numpy as np

class HyperParams:
    def __init__(self, arg_hyps=None):
        
        hyp_dict = dict()
        hyp_dict['string_hyps'] = {
                    "exp_name":"default",
                    "model_type":"gru", # Options include 'dense', 'conv', 'a3c'
                    "env_type":"Pong-v0", 
                    "optim_type":'rmsprop' # Options: rmsprop, adam
                    }
        hyp_dict['int_hyps'] = {
                    "n_epochs": 3, # PPO update epoch count
                    "batch_size": 256, # PPO update batch size
                    "max_tsteps": int(1e6),
                    "n_tsteps": 128, # Maximum number of tsteps per rollout per perturbed copy
                    "n_envs": 11, # Number of parallel python processes
                    "n_frame_stack":3, # Number of frames to stack in MDP state
                    "n_rollouts": 22,
                    'h_size':288,
                    "n_past_rews":25,
                    "grid_size": 15,
                    "unit_size": 4,
                    "n_foods": 2,
                    }
        hyp_dict['float_hyps'] = {
                    "lr":0.0001,
                    "lr_low": float(1e-12),
                    "lambda_":.95,
                    "gamma":.99,
                    "val_const":1,
                    "entr_coef":.01,
                    "entr_coef_low":.001,
                    "max_norm":.5,
                    "epsilon": .2, # PPO update clipping constant
                    "epsilon_low":.05,
                    }
        hyp_dict['bool_hyps'] = {
                    "resume":False,
                    "render": False,
                    "clip_vals": False,
                    "decay_eps": False,
                    "decay_lr": False,
                    "decay_entr": False,
                    "use_nstep_rets": False,
                    "norm_advs": True,
                    "norm_batch_advs": False,
                    "use_bnorm": False,
                    }
        self.hyps = self.read_command_line(hyp_dict)
        if arg_hyps is not None:
            for arg_key in arg_hyps.keys():
                self.hyps[arg_key] = arg_hyps[arg_key]

        # Hyperparameter Manipulations
        self.hyps['grid_size'] = [self.hyps['grid_size'],self.hyps['grid_size']]
        if self.hyps['batch_size'] > self.hyps['n_rollouts']*self.hyps['n_tsteps']:
            self.hyps['batch_size'] = self.hyps['n_rollouts']*self.hyps['n_tsteps']

        # Model Type
        model_type = self.hyps['model_type'].lower()
        if "conv" == model_type:
            self.hyps['model'] = ConvModel
        elif "a3c" == model_type:
            self.hyps['model'] = A3CModel
        elif "fc" == model_type or "dense" == model_type:
            self.hyps['model'] = FCModel
        elif "gru" == model_type:
            self.hyps['model'] = GRUModel
        else:
            self.hyps['model'] = ConvModel

        # Preprocessor Type
        env_type = self.hyps['env_type'].lower()
        if "pong" in env_type:
            self.hyps['preprocess'] = preprocessing.pong_prep
        elif "breakout" in env_type:
            self.hyps['preprocess'] = preprocessing.breakout_prep
        elif "snake" in env_type:
            self.hyps['preprocess'] = preprocessing.snake_prep
        else:
            self.hyps['preprocess'] = preprocessing.atari_prep

    def read_command_line(self, hyps_dict):
        """
        Reads arguments from the command line. If the parameter name is not declared in __init__
        then the command line argument is ignored.
    
        Pass command line arguments with the form parameter_name=parameter_value
    
        hyps_dict - dictionary of hyperparameter dictionaries with keys:
                    "bool_hyps" - dictionary with hyperparameters of boolean type
                    "int_hyps" - dictionary with hyperparameters of int type
                    "float_hyps" - dictionary with hyperparameters of float type
                    "string_hyps" - dictionary with hyperparameters of string type
        """
        
        bool_hyps = hyps_dict['bool_hyps']
        int_hyps = hyps_dict['int_hyps']
        float_hyps = hyps_dict['float_hyps']
        string_hyps = hyps_dict['string_hyps']
        
        if len(sys.argv) > 1:
            for arg in sys.argv:
                arg = str(arg)
                sub_args = arg.split("=")
                if sub_args[0] in bool_hyps:
                    bool_hyps[sub_args[0]] = sub_args[1] == "True"
                elif sub_args[0] in float_hyps:
                    float_hyps[sub_args[0]] = float(sub_args[1])
                elif sub_args[0] in string_hyps:
                    string_hyps[sub_args[0]] = sub_args[1]
                elif sub_args[0] in int_hyps:
                    int_hyps[sub_args[0]] = int(sub_args[1])
    
        return {**bool_hyps, **float_hyps, **int_hyps, **string_hyps}

# Methods

def hyper_search(hyps, hyp_ranges, keys, idx, trainer, search_log):
    """
    hyps - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of ranges for hyperparameters to take over the search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list of values to search over for that hyperparameter
    keys - keys of the hyperparameters to be searched over. Used to
            allow order of hyperparameter search
    idx - the index of the current key to be searched over
    trainer - trainer object that handles training of model
    """
    if idx >= len(keys):
        if 'search_id' not in hyps:
            hyps['search_id'] = 0
            hyps['exp_name'] = hyps['exp_name']+"0"
            hyps['hyp_search_count'] = np.prod([len(hyp_ranges[key]) for key in keys])
        id_ = len(str(hyps['search_id']))
        hyps['search_id'] += 1
        hyps['exp_name'] = hyps['exp_name'][:-id_]+str(hyps['search_id'])
        best_avg_rew = trainer.train(hyps)
        params = [str(key)+":"+str(hyps[key]) for key in keys]
        search_log.write(", ".join(params)+" – BestRew:"+str(best_avg_rew)+"\n")
        search_log.flush()
    else:
        key = keys[idx]
        for param in hyp_ranges[key]:
            hyps[key] = param
            hyper_search(hyps, hyp_ranges, keys, idx+1, trainer, search_log)
    return

def make_hyper_range(low, high, range_len, method="log"):
    if method.lower() == "random":
        param_vals = np.random.random(low, high+1e-5, size=range_len)
    elif method.lower() == "uniform":
        step = (high-low)/(range_len-1)
        pos_step = (step > 0)
        range_high = high+(1e-5)*pos_step-(1e-5)*pos_step
        param_vals = np.arange(low, range_high, step=step)
    else:
        range_low = np.log(low)/np.log(10)
        range_high = np.log(high)/np.log(10)
        step = (range_high-range_low)/(range_len-1)
        arange = np.arange(range_low, range_high, step=step)
        if len(arange) < range_len:
            arange = np.append(arange, [range_high])
        param_vals = 10**arange
    param_vals = [float(param_val) for param_val in param_vals]
    return param_vals
