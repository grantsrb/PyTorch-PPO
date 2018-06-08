import sys

class HyperParams:
    def __init__(self):
        
        hyp_dict = dict()
        hyp_dict['string_hyps'] = {
                    "exp_name":"default",
                    "model_type":"conv", # Options include 'dense', 'conv', 'a3c'
                    "env_type":"Pong-v0", 
                    "optim_type":'adam' # Options: rmsprop, adam
                    }
        hyp_dict['int_hyps'] = {
                    "n_epochs": 3, # PPO update epoch count
                    "batch_size": 256, # PPO update batch size
                    "max_tsteps": int(1e6),
                    "n_tsteps": 128, # Maximum number of tsteps per rollout per perturbed copy
                    "n_envs": 11, # Number of parallel python processes
                    "n_frame_stack":2, # Number of frames to stack in MDP state
                    "emb_size":200, # Size of hidden layer in "small" network
                    "rand_seed":1,
                    "n_rollouts":32,
                    "grid_size": 15,
                    "unit_size": 4,
                    "n_foods": 2,
                    "cache_size": 0,
                    "n_past_rews":25,
                    }
        hyp_dict['float_hyps'] = {
                    "lr":0.001,
                    "lr_low": float(1e-12),
                    "lambda_":.95,
                    "gamma":.99,
                    "sigma":.05,
                    "sigma_low":2e-7,
                    "val_const":.1,
                    "entr_coef":.01,
                    "entr_coef_low":.001,
                    "max_norm":.5,
                    "epsilon": .2, # PPO update clipping constant
                    "epsilon_low":.05,
                    }
        hyp_dict['bool_hyps'] = {
                    "bnorm":False,
                    "resume":False,
                    "render": False,
                    "clip_vals": False,
                    "decay_eps": True,
                    "decay_lr": True,
                    "decay_entr": True,
                    "collector_cuda": True,
                    "use_nstep_rets": False,
                    "norm_advs": False,
                    "norm_batch_advs": True,
                    "use_bnorm": False,
                    "eval_vals": False,
                    }
        self.hyps = self.read_command_line(hyp_dict)

        self.hyps['grid_size'] = [self.hyps['grid_size'],self.hyps['grid_size']]
        if self.hyps['batch_size'] > self.hyps['n_rollouts']*self.hyps['n_tsteps']:
            self.hyps['batch_size'] = self.hyps['n_rollouts']*self.hyps['n_tsteps']

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
