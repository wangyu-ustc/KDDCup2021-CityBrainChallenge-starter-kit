import torch.nn as nn
import torch
import numpy as np

from torch.nn import functional as F



def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)


def t(x): return torch.from_numpy(x).float()

def list_to_tensor(x):
    x = np.array(x)
    return t(x)



# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X.detach())



# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X.detach())



def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


def policy_loss(old_log_prob,log_prob,advantage,eps):
    loss = {}
    for key in old_log_prob.keys():
        loss[key] = policy_loss_single(old_log_prob[key],log_prob[key],advantage[key],eps)
    return loss

def policy_loss_single(old_log_prob, log_prob, advantage, eps):

    # print("this is log_prob",log_prob)
    # print("this is old_log_prob",old_log_prob)
    #ratio = (log_prob - old_log_prob.detach()).exp()
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

    m = torch.min(ratio * advantage, clipped)
    return -m


def get_old(log_prob):
    #print(type(log_prob))
    new = {}
    for key in log_prob.keys():
        new[key] = log_prob[key].detach()
    return new



