import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2grey

class Model(nn.Module):
    def __init__(self, input_space, output_space, emb_size=256, bnorm=False):
        super(Model, self).__init__()
        self.input_space = input_space
        self.output_space = output_space
        self.emb_size = emb_size
        self.use_bnorm = bnorm

        # Embedding Net
        self.convs = nn.ModuleList([])
        shape = input_space.copy()

        self.conv1 = self.conv_block(input_space[-3], 16, ksize=7, stride=1, padding=0, bnorm=bnorm, activation='elu')
        self.convs.append(self.conv1)
        shape = self.new_shape(shape, 16, ksize=7, stride=1, padding=0)

        self.conv2 = self.conv_block(16, 32, ksize=3, stride=1, padding=0, bnorm=bnorm, activation='elu')
        self.convs.append(self.conv2)
        shape = self.new_shape(shape, 32, ksize=3, stride=1, padding=0) 

        self.conv3 = self.conv_block(32, 32, ksize=3, stride=1, padding=0,bnorm=bnorm, activation='elu')
        self.convs.append(self.conv3)
        shape = self.new_shape(shape, 32, ksize=3, stride=1, padding=0)

        self.conv4 = self.conv_block(32, 32, ksize=3, stride=2, padding=0,bnorm=bnorm, activation='elu')
        self.convs.append(self.conv4)
        shape = self.new_shape(shape, 32, ksize=3, stride=2, padding=0)

        self.conv5 = self.conv_block(32, 32, ksize=3, stride=2,padding=0,bnorm=bnorm, activation='elu')
        self.convs.append(self.conv5)
        shape = self.new_shape(shape, 32, ksize=3, stride=2, padding=0)

        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        print("Features Flat Size:", self.flat_size)
        self.proj_matrx = nn.Linear(self.flat_size, self.emb_size)

        # Policy
        self.emb_bnorm = nn.BatchNorm1d(self.emb_size)
        self.pi = nn.Linear(self.emb_size, self.output_space)
        self.value = nn.Linear(self.emb_size, 1)

        # Inverse Dynamics
        self.inv_dyn1 = nn.Linear(self.emb_size, 256)
        self.inv_dyn2 = nn.Linear(self.emb_size, 256)
        self.inv_dyn3 = nn.Linear(256, self.output_space)

    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def new_shape(self, shape, depth, ksize=3, padding=1, stride=2):
        shape[-1] = self.new_size(shape[-1], ksize=ksize, padding=padding, stride=stride)
        shape[-2] = self.new_size(shape[-2], ksize=ksize, padding=padding, stride=stride)
        shape[-3] = depth
        return shape

    def forward(self, x):
        embs = self.encoder(x)
        val, pi = self.policy(embs)
        #return val, pi, embs
        return val, pi

    def encoder(self, state):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor with shape (BatchSize, Channels, Height, Width)
        """
        feats = self.features(state)
        feats = feats.view(feats.shape[0], -1)
        feats = self.proj_matrx(feats)
        return feats

    def policy(self, state_emb, bnorm=True):
        """
        Uses the state embedding to produce an action.

        state_emb - the state embedding created by the encoder
        """
        if self.use_bnorm:
            state_emb = self.emb_bnorm(state_emb)
        pi = self.pi(state_emb)
        value = self.value(Variable(state_emb.data))
        return value, pi

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
        if activation is not None: 
            activation=activation.lower()
        if "relu" == activation:
            block.append(nn.ReLU())
        elif "selu" == activation:
            block.append(nn.SELU())
        elif "elu" == activation:
            block.append(nn.ELU())
        elif "tanh" == activation:
            block.append(nn.Tanh())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="relu", bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None: activation=activation.lower()
        if "relu" == activation:
            block.append(nn.ReLU())
        elif "selu" == activation:
            block.append(nn.SELU())
        elif "elu" == activation:
            block.append(nn.ELU())
        elif "tanh" == activation:
            block.append(nn.Tanh())
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

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

    @staticmethod
    def preprocess(pic, env_type='snake-v0'):
        if env_type == "Pong-v0":
            pic = pic[35:195] # crop
            pic = pic[::2,::2,0] # downsample by factor of 2
            pic[pic == 144] = 0 # erase background (background type 1)
            pic[pic == 109] = 0 # erase background (background type 2)
            pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
        elif 'Breakout' in env_type:
            pic = pic[35:] # crop
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

class FwdDynamics(nn.Module):
    def __init__(self, emb_size, action_space, bnorm=False):
        super(FwdDynamics, self).__init__()
        self.emb_size = emb_size
        self.action_space = action_space
        self.use_bnorm = bnorm

        # Forward Dynamics
        self.fwd_dyn1 = nn.Linear(self.emb_size, 256)
        eye = torch.eye(self.action_space).float()
        if torch.cuda.is_available(): eye = eye.cuda()
        self.action_one_hots = Variable(eye)
        self.action_layer = nn.Linear(self.action_space, 256)
        self.fwd_dyn2 = nn.Linear(256, self.emb_size)

    def forward(self, h, a, bnorm=False):
        """
        Forward dynamics model predicts the embedding of the next state.

        h - Variable FloatTensor of current state embedding
        a - list of action indices as ints or a LongTensor of the actual actions taken
        """
        print(type(a))
        print(a.shape)

        actions = self.action_one_hots[a]
        mid = F.elu(self.fwd_dyn1(h)+self.action_layer(actions))
        pred = self.fwd_dyn2(mid)
        return pred
