from hyperparams import HyperParams, hyper_search, make_hyper_range
from ppo import PPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = PPO()
    hyper_params = HyperParams()
    hyps = hyper_params.hyps
    hyp_ranges = {
                "gamma": [.98, .99, .995],
                "lambda_": [.94, .95, .96, .97], 
                "lr": make_hyper_range(5e-5,1e-3,5,"log"),
                "val_const": [.005, .01, .1, 1],
                "entr_coef": [7e-3, .02] 
                }
    keys = list(hyp_ranges.keys())
    hyps['env_type'] = "Breakout-v0"
    hyps['exp_name'] = "brkoutlongrolls"
    hyps['n_tsteps'] = 256
    hyps['n_rollouts'] = 11
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 3000000

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, ppo_trainer)

