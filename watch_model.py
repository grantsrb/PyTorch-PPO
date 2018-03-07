import sys
import torch
from torch.autograd import Variable
import gym
import numpy as np
import conv_model
import dense_model



file_name = str(sys.argv[1])
env_type = 'BreakoutNoFrameskip-v4'
model_type = 'dense'

# Environment Choices
grid_size = [15,15]
n_foods = 2
unit_size = 4
if model_type == 'dense':
    n_frame_stack = 2 # number of observations to stack for a single environment state
else:
    n_frame_stack = 4 # number of observations to stack for a single environment state


if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        str_arg = str(arg)
        if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
        if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
        if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
        if "n_frame_stack=" in str_arg: n_frame_stack= int(str_arg[len('n_frame_stack='):])
        if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]
        if "model_type=" in str_arg: model_type = str_arg[len('model_type='):]

print("file_name:", file_name)
print("n_frame_stack:", n_frame_stack)
print("grid_size:", grid_size)
print("n_foods:", n_foods)
print("unit_size:", unit_size)
print("env_type:", env_type)
print("model_type:", model_type)

if model_type == "dense":
    Model = dense_model.Model
else:
    Model = conv_model.Model


preprocess = Model.preprocess
env_type = env_type
env = gym.make(env_type)
env.grid_size = grid_size
env.n_foods = n_foods
env.unit_size = unit_size
action_space = env.action_space.n
if env_type == 'Pong-v0':
    action_space = 2
elif 'Breakout' in env_type:
    action_space = 2

def get_action( pi):
    """
    Stochastically selects an action based on the action probabilities.

    pi - torch FloatTensor of the raw action prediction
    """

    action_ps = softmax(pi.numpy()).squeeze()
    action = np.random.choice(action_space, p=action_ps)
    return int(action)

def make_state( prepped_obs, prev_state=None):
    """
    Combines the new, prepprocessed observation with the appropriate parts of the previous
    state to make a new state that is ready to be passed through the network. If prev_state
    is None, the state is filled with zeros outside of the new observation.

    prepped_obs - torch FloatTensor of peprocessed observation
    prev_state - torch FloatTensor of previous environment state
    """

    if prev_state is None:
        prev_state = np.zeros(state_shape, dtype=np.float32)

    next_state = np.concatenate([prepped_obs, prev_state[:-prepped_obs.shape[0]]], axis=0)
    return next_state

def next_state( env, prev_state, obs, reset):
    """
    Get the next state of the environment.

    env - environment of interest
    prev_state - ndarray of the state used in the most recent action
                prediction
    obs - ndarray returned from the most recent step of the environment
    reset - boolean denoting the reset signal from the most recent step of the
            environment
    """
    if reset:
        obs = env.reset()
        prev_state = None
    prepped_obs = preprocess(obs, env_type)
    state = make_state(prepped_obs, prev_state)
    return state

def softmax( X, theta=1.0, axis=-1):
    """
    * Inspired by https://nolanbconaway.github.io/blog/2017/softmax-numpy *

    Computes the softmax of each element along an axis of X.

    X - ndarray of at least 2 dimensions
    theta - float used as a multiplier prior to exponentiation
    axis - axis to compute values along

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    X = X * float(theta)
    X = X - np.expand_dims(np.max(X, axis = axis), axis)
    X = np.exp(X)
    ax_sum = np.expand_dims(np.sum(X, axis = axis), axis)
    p = X / ax_sum
    return p
observation = env.reset()
prepped_obs = preprocess(observation, env_type)
obs_shape = observation.shape
prepped_shape = prepped_obs.shape
state_shape = [n_frame_stack*prepped_shape[0],*prepped_shape[1:]]
state = make_state(prepped_obs)
net = Model(state_shape, action_space, env_type=env_type)
net = net
dummy = Variable(torch.ones(2,*state_shape))
net.forward(dummy)
net.load_state_dict(torch.load(file_name))
net.train(mode=False)
net.req_grads(False)

lives = 12
last_reset = 0
ep_reward = 0
counter = 0
while True:
    counter+=1
    value, pi = net.forward(Variable(torch.FloatTensor(state.copy()).unsqueeze(0)))
    action = get_action(pi.data)

    obs, reward, done, info = env.step(action+2*(env_type=='Pong-v0')+2*('Breakout' in env_type))
    ep_reward += reward
    env.render()
    last_reset+=1
    if lives > 2*info['ale.lives'] or last_reset>100:
        obs, reward, new_done, info = env.step(1)
        if lives > 2*info['ale.lives']:
            lives -= 1
        last_reset = 0
        done = new_done or done
    if done:
        lives = 12
        print("done", ep_reward)
        ep_reward=0
    state = next_state(env, state, obs, done)
    if counter > 16:
        print("counter up")
        counter = 0
