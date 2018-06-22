from hyperparams import HyperParams, hyper_search, make_hyper_range
from ppo import PPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = PPO()
    hyps = dict()
    hyp_ranges = {
                "lr": [9.5e-5, 1e-4, 1.5e-4],
                "val_const": [.005, .0075, .01],
                }
    keys = list(hyp_ranges.keys())
    hyps['lambda_'] = .93
    hyps['gamma'] = .985
    hyps['entr_coef'] = .008
    hyps['env_type'] = "Breakout-v0"
    hyps['exp_name'] = "brkout2"
    hyps['n_tsteps'] = 256
    hyps['n_rollouts'] = 11
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 5000000
    hyps['n_frame_stack'] = 3
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, ppo_trainer, search_log)
    search_log.close()

