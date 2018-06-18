#!/bin/bash

rm -rf ~/.nv/
killall python3

python3 main.py exp_name=bigval resume=False env_type=Breakout-v0 optim_type=adam model_type=conv n_frame_stack=3 norm_advs=False norm_batch_advs=True use_nstep_rets=False clip_vals=False decay_eps=True decay_lr=False decay_entr=False incr_gamma=False use_bnorm=False n_envs=11 n_rollouts=12 n_tsteps=256 batch_size=256 n_epochs=3 max_tsteps=20000000 max_norm=.5 val_const=.1 epsilon=.17 epsilon_low=.13 entr_coef=.01 entr_coef_low=.007 lr=.0001 lr_low=1e-5 lambda_=.95 gamma=.99 gamma_high=.995

