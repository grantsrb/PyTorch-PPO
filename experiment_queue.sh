#!/bin/bash

python3 train_model.py exp_name=dataconv model_type=conv n_frame_stack=4 norm_advs=False use_nstep_rets=True norm_batch_advs=True clip_vals=True decay_eps=True epsilon=.2 n_tsteps=16 n_envs=16 n_rollouts=120 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=10000000 max_norm=0.05 val_const=1 entropy_const=.005 

python3 train_model.py exp_name=bnormconv model_type=conv n_frame_stack=4 norm_advs=False use_nstep_rets=True norm_batch_advs=True clip_vals=True decay_eps=True use_bnorm=True epsilon=.2 n_tsteps=16 n_envs=16 n_rollouts=120 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=10000000 max_norm=0.05 val_const=1 entropy_const=.005 


python3 train_model.py exp_name=dataconvmaxnorm model_type=conv n_frame_stack=4 norm_advs=False use_nstep_rets=True norm_batch_advs=True clip_vals=True decay_eps=True epsilon=.2 n_tsteps=16 n_envs=16 n_rollouts=120 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=10000000 max_norm=0.5 val_const=1 entropy_const=.005 

python3 train_model.py exp_name=dataconvtsteps model_type=conv n_frame_stack=4 norm_advs=False use_nstep_rets=True norm_batch_advs=True clip_vals=True decay_eps=True epsilon=.2 n_tsteps=32 n_envs=16 n_rollouts=120 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=10000000 max_norm=0.05 val_const=1 entropy_const=.01 
