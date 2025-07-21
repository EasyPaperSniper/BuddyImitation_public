import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal
import einops


def build_mlp(input_dim, output_dim, hidden_dims, activation):
    actor_layers = []
    actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
    actor_layers.append(activation)
    for layer_index in range(len(hidden_dims)):
        if layer_index == len(hidden_dims) - 1:
            actor_layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
        else:
            actor_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            actor_layers.append(activation)
    return actor_layers




def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)
    


class MANN_network(nn.Module):
    def __init__(self, obs_dim, nxt_obs_dim, lat_dim, num_experts, hg, h, activation):
        super(MANN_network, self).__init__()
        self.gatingNN = nn.Sequential(*build_mlp(obs_dim+lat_dim, num_experts, [hg]*3, activation))
        self.motionNN = [
            (
                nn.Parameter(torch.empty(num_experts, obs_dim+lat_dim, h)),
                nn.Parameter(torch.empty(num_experts,  h)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, h+lat_dim, h)),
                nn.Parameter(torch.empty(num_experts,  h)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, h+lat_dim, nxt_obs_dim)),
                nn.Parameter(torch.empty(num_experts,  nxt_obs_dim)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.motionNN):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)



    def forward(self, x, z ):
        coefficients = F.softmax(self.gatingNN(torch.cat((x, z), dim=1)), dim=1)
        layer_out = x

        for (weight, bias, activation) in self.motionNN:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((layer_out, z), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out
        
        return layer_out
