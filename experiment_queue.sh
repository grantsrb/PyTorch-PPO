#!/bin/bash

python3 entry.py exp_name=dense norm_advs=False norm_batch_advs=False clip_vals=True norm_returns=False decay_eps=False epsilon=.15 n_tsteps=16 n_envs=16 n_rollouts=64 batch_size=128 n_epochs=4 env_type=BreakoutNoFrameskip-v4 max_tsteps=4000000

python3 entry.py exp_name=densefullnorm norm_advs=True norm_batch_advs=False clip_vals=True norm_returns=False decay_eps=False epsilon=.15 n_tsteps=16 n_envs=16 n_rollouts=64 batch_size=128 n_epochs=4 env_type=BreakoutNoFrameskip-v4 max_tsteps=4000000

python3 entry.py exp_name=densenormbatch norm_advs=False norm_batch_advs=True clip_vals=True norm_returns=False decay_eps=False epsilon=.15 n_tsteps=16 n_envs=16 n_rollouts=64 batch_size=128 n_epochs=4 env_type=BreakoutNoFrameskip-v4 max_tsteps=4000000

python3 entry.py exp_name=conv resume=True model_type=conv norm_advs=False norm_batch_advs=False clip_vals=True norm_returns=False decay_eps=False epsilon=.15 n_tsteps=16 n_envs=16 n_rollouts=64 batch_size=128 n_epochs=4 env_type=BreakoutNoFrameskip-v4 max_tsteps=4000000 n_frame_stack=4
