#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 train_model.py resume=False exp_name=conv model_type=conv n_frame_stack=4 norm_advs=True norm_batch_advs=False use_nstep_rets=True clip_vals=True decay_eps=True decay_lr=True use_bnorm=False epsilon=.20 n_tsteps=128 n_envs=8 n_rollouts=16 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=20000000 max_norm=.5 val_const=.75 entropy_const=.01 bootstrap_next=True lr=1e-4 lambda_=.95 gamma=.99 fresh_advs=False

CUDA_VISIBLE_DEVICES=0 python3 train_model.py resume=False exp_name=batchAdvs model_type=conv n_frame_stack=4 norm_advs=False norm_batch_advs=True use_nstep_rets=True clip_vals=True decay_eps=True decay_lr=True use_bnorm=False epsilon=.20 n_tsteps=256 n_envs=8 n_rollouts=16 batch_size=128 n_epochs=4 env_type=Pong-v0 max_tsteps=20000000 max_norm=.5 val_const=1 entropy_const=.01 bootstrap_next=True lr=1e-4 lambda_=.95 gamma=.99 fresh_advs=False
