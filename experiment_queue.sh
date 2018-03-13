#!/bin/bash


python3 train_model.py resume=False exp_name=conv model_type=conv n_frame_stack=4 norm_advs=True norm_batch_advs=False use_nstep_rets=True clip_vals=True decay_eps=True decay_lr=True use_bnorm=True epsilon=.20 n_tsteps=256 n_envs=16 n_rollouts=16 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=20000000 max_norm=.4 val_const=.75 entropy_const=.001 bootstrap_next=True lr=1e-3 lambda_=.95 gamma=.99 fresh_advs=False

python3 train_model.py resume=True exp_name=testing model_type=dense n_frame_stack=2 norm_advs=True norm_batch_advs=False use_nstep_rets=True clip_vals=True decay_eps=True decay_lr=True use_bnorm=True epsilon=.10 n_tsteps=256 n_envs=16 n_rollouts=16 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=10000000 max_norm=.4 val_const=.75 entropy_const=.001 bootstrap_next=True lr=1e-3 lambda_=.95 gamma=.99 fresh_advs=False
