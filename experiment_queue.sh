#!/bin/bash

rm -rf ~/.nv/

python3 main.py exp_name=reluproj2 resume=False env_type=Pong-v0 model_type=conv n_frame_stack=3 norm_advs=True norm_batch_advs=False use_nstep_rets=False clip_vals=False decay_eps=True decay_lr=True decay_entr=True use_bnorm=False epsilon=.2 epsilon_low=.1 n_tsteps=64 n_envs=9 n_rollouts=44 batch_size=128 n_epochs=4 max_tsteps=40000000 max_norm=.1 val_const=1 entr_coef_high=.01 entr_coef_low=.0009 lr=1e-4 lr_low=1e-11 lambda_=.95 gamma=.995

