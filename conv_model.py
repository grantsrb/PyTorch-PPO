import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2grey

'''
Simple, sequential convolutional net.
'''

class Model(nn.Module):
    def __init__(self, input_space, output_space, emb_size=256, env_type='snake-v0', bnorm=False):
        super(Model, self).__init__()

        self.input_space = input_space
        self.output_space = output_space
        self.emb_size = emb_size
        self.env_type = env_type

        self.convs = nn.ModuleList([])

        # Embedding Net
        shape = input_space.copy()
        self.conv1 = self.conv_block(input_space[-3],32, stride=2, bnorm=bnorm, activation='elu')
        self.convs.append(self.conv1)
        shape[-1] = self.new_size(shape[-1], ksize=3, padding=1, stride=2)
        shape[-2] = self.new_size(shape[-2], ksize=3, padding=1, stride=2)
        shape[-3] = 32
        self.conv2 = self.conv_block(32, 32, stride=2,bnorm=bnorm, activation='elu')
        self.convs.append(self.conv2)
        shape[-1] = self.new_size(shape[-1], ksize=3, padding=1, stride=2)
        shape[-2] = self.new_size(shape[-2], ksize=3, padding=1, stride=2)
        shape[-3] = 32
        self.conv3 = self.conv_block(32, 32, stride=2,bnorm=bnorm, activation='elu')
        self.convs.append(self.conv3)
        shape[-1] = self.new_size(shape[-1], ksize=3, padding=1, stride=2)
        shape[-2] = self.new_size(shape[-2], ksize=3, padding=1, stride=2)
        shape[-3] = 32
        self.conv4 = self.conv_block(32, 32, stride=2, bnorm=bnorm, activation='elu')
        self.convs.append(self.conv4)
        shape[-1] = self.new_size(shape[-1], ksize=3, padding=1, stride=2)
        shape[-2] = self.new_size(shape[-2], ksize=3, padding=1, stride=2)
        shape[-3] = 32
        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        self.resize_emb = nn.Linear(self.flat_size, self.emb_size)

        # Policy
        self.emb_bnorm = nn.BatchNorm1d(self.emb_size)
        self.pi = self.dense_block(self.emb_size, self.output_space, activation='none', bnorm=False)
        self.value = self.dense_block(self.emb_size, 1, activation='none', bnorm=False)

        # Forward Dynamics
        self.fwd_dyn1 = nn.Linear(self.emb_size, 256)
        self.action_one_hots = torch.zeros(self.output_space, self.output_space).float()
        for i in range(self.action_one_hots.shape[0]):
            self.action_one_hots[i,i] = 1
        self.action_one_hots = Variable(self.action_one_hots)
        self.action_layer = nn.Linear(self.output_space, 256)
        self.fwd_dyn2 = nn.Linear(256, self.emb_size)

        # Inverse Dynamics
        self.inv_dyn1 = nn.Linear(self.emb_size, 256)
        self.inv_dyn2 = nn.Linear(self.emb_size, 256)
        self.inv_dyn3 = nn.Linear(256, self.output_space)

    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def forward(self, x, bnorm=False):
        embs = self.emb_net(x)
        val, pi = self.policy(embs, bnorm=bnorm)
        return val, pi

    def emb_net(self, state):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor with shape (BatchSize, Channels, Height, Width)
        """
        feats = self.features(state)
        feats = feats.view(feats.shape[0], -1)
        state_embs = self.resize_emb(feats)
        return state_embs

    def policy(self, state_emb, bnorm=True):
        """
        Uses the state embedding to produce an action.

        state_emb - the state embedding created by the emb_net
        """
        if bnorm:
            state_emb = self.emb_bnorm(state_emb)
        pi = self.pi(state_emb)
        value = self.value(Variable(state_emb.data))
        return value, pi

    def fwd_dynamics(self, h, a, bnorm=False):
        """
        Forward dynamics model predicts the embedding of the next state.

        h - Variable FloatTensor of current state embedding
        a - list of action indices as ints or a LongTensor of the actual actions taken
        """

        actions = self.action_one_hots[a]
        mid = F.relu(self.fwd_dyn1(h)+self.action_layer(actions))
        pred = self.fwd_dyn2(mid)
        return pred

    def inv_dynamics(self, h1, h2):
        """
        Predicts the action between two consequtive state embeddings.

        h1 - Variable FloatTensor of the current state embedding
        h2 - Variable FloatTensor of the next state embedding
        """

        intmd = F.elu(self.inv_dyn1(h1)+self.inv_dyn2(h2))
        action = self.inv_dyn3(intmd)
        return action

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1, padding=1, activation="relu", max_pool=False, bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride, padding=padding))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "selu" in activation:
            block.append(nn.SELU())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="relu", bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "selu" in activation:
            block.append(nn.SELU())
        if bnorm:
            block.append(nn.BatchNorm1d(chan_out))
        return nn.Sequential(*block)

    def add_noise(self, x, mean=0.0, std=0.01):
        """
        Adds a normal distribution over the entries in a matrix.
        """

        means = torch.zeros(*x.size()).float()
        if mean != 0.0:
            means = means + mean
        noise = torch.normal(means,std=std)
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x+noise

    def multiply_noise(self, x, mean=1, std=0.01):
        """
        Multiplies a normal distribution over the entries in a matrix.
        """

        means = torch.zeros(*x.size()).float()
        if mean != 0:
            means = means + mean
        noise = torch.normal(means,std=std)
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x*noise

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        calc_bool - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

    @staticmethod
    def preprocess(pic, env_type='snake-v0'):
        if env_type == "Pong-v0":
            pic = pic[35:195] # crop
            pic = pic[::2,::2,0] # downsample by factor of 2
            pic[pic == 144] = 0 # erase background (background type 1)
            pic[pic == 109] = 0 # erase background (background type 2)
            pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
        elif 'Breakout' in env_type:
            pic = pic[35:195] # crop
            pic = rgb2grey(pic)
            pic = resize(pic, (52,52), mode='constant')
            pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
        elif env_type == "snake-v0":
            new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
            new_pic[:,:][pic[:,:,0]==1] = 1
            new_pic[:,:][pic[:,:,0]==255] = 1.5
            new_pic[:,:][pic[:,:,1]==255] = 0
            new_pic[:,:][pic[:,:,2]==255] = .33
            pic = new_pic
        return pic[None]
