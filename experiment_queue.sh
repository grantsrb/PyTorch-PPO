#!/bin/bash

python3 entry.py exp_name=dense norm_advs=False norm_batch_advs=False clip_vals=True decay_eps=True epsilon=.2 n_tsteps=128 n_envs=16 n_rollouts=16 batch_size=256 n_epochs=4 env_type=Pong-v0 max_tsteps=4000000 max_norm=0.05 val_const=1 entropy_const=.005

python3 entry.py exp_name=densefullnorm norm_advs=True norm_batch_advs=False clip_vals=True decay_eps=True epsilon=.2 n_tsteps=128 n_envs=16 n_rollouts=16 batch_size=256 n_epochs=4 env_type=Pong-v0 max_tsteps=4000000 max_norm=0.05 val_const=1 entropy_const=.005

python3 entry.py exp_name=densenormbatch norm_advs=False norm_batch_advs=True clip_vals=True decay_eps=True epsilon=.2 n_tsteps=128 n_envs=16 n_rollouts=16 batch_size=256 n_epochs=4 env_type=Pong-v0 max_tsteps=4000000 max_norm=0.05 val_const=1 entropy_const=.005

