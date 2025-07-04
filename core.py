from abc import abstractmethod
from typing import Any, Iterable, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes:list[int], activation:nn.Module, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module:nn.Module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20
NUM_LAYERS = 2

class ActorBase(nn.Module):
    @abstractmethod
    def __init__(self, state_dim:int, price_dim:int, act_dim:int, hidden_sizes:list[int], activation:Any, act_limit:tuple[float, float]):
        """
        Initialize the actor network.
        Parameters:
            state_dim (int): Dimension of the state input.
            price_dim (int): Dimension of the price input.
            act_dim (int): Dimension of the action output.
            hidden_sizes (list[int]): List of hidden layer sizes.
            activation (Any): Activation function to use in the network.
            act_limit (tuple[float, float]): Action limits for scaling the output.
        """
        super().__init__()
    
    @abstractmethod
    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Forward pass of the actor network.
        
        Parameters:
            obs (dict): Observation dictionary containing 'net' and 'price'.
            deterministic (bool): If True, returns deterministic actions.
            with_logprob (bool): If True, returns log probabilities of actions.
        
        Returns:
            pi_action (torch.Tensor): The action to take.
            logp_pi (torch.Tensor or None): Log probability of the action if with_logprob is True.
        """

class QFunctionBase(nn.Module):
    @abstractmethod
    def __init__(self, state_dim:int, price_dim:int, act_dim:int, hidden_sizes:list[int], activation:Any):
        """
        Initialize the Q-function network.
        
        Parameters:
            state_dim (int): Dimension of the state input.
            price_dim (int): Dimension of the price input.
            act_dim (int): Dimension of the action input.
            hidden_sizes (list[int]): List of hidden layer sizes.
            activation (Any): Activation function to use in the network.
        """
        super().__init__()
    
    @abstractmethod
    def forward(self, obs, act):
        """
        Forward pass of the Q-function network.
        
        Parameters:
            obs (dict): Observation dictionary containing 'net' and 'price'.
            act (torch.Tensor): Action tensor.
        
        Returns:
            q_value (torch.Tensor): The Q-value for the given observation and action.
        """

class ActorLSTM(ActorBase):
    def __init__(self, state_dim:int, price_dim:int, act_dim:int, hidden_sizes:list[int], activation:Any, act_limit:tuple[float, float]):
        super().__init__(state_dim, price_dim, act_dim, hidden_sizes, activation, act_limit)
        hidden_size = hidden_sizes[0]
        hidden_size_last = hidden_sizes[-1]
        self.state_net = nn.LSTM(input_size = state_dim, hidden_size = hidden_size, num_layers = NUM_LAYERS, batch_first=True)
        self.price_net = nn.Linear(price_dim, hidden_size)
        self.net = mlp([hidden_size * 2] + hidden_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_size_last, act_dim)
        self.log_std_layer = nn.Linear(hidden_size_last, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # state input is (N, L, Hin), price input is (N, Hprice)
        state_feature, (hn, cn) = self.state_net(obs["net"]) 
        price_feature = self.price_net(obs["price"])
        # Concatenate the last hidden state of LSTM and price feature
        obs = torch.cat([hn[-1], price_feature], dim=-1)  # (Hout,)
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit[0] + 0.5 * (pi_action + 1.0) * (self.act_limit[1] - self.act_limit[0])

        return pi_action, logp_pi


class QFunctionLSTM(QFunctionBase):
    def __init__(self, state_dim:int, price_dim:int, act_dim:int, hidden_sizes:list[int], activation:Any):
        super().__init__(state_dim, price_dim, act_dim, hidden_sizes, activation)
        hidden_size = hidden_sizes[0]
        self.state_net = nn.LSTM(input_size = state_dim, hidden_size = hidden_size, num_layers = NUM_LAYERS, batch_first=True)
        self.price_net = nn.Linear(price_dim, hidden_size)
        self.q = mlp([hidden_size * 2 + act_dim] + hidden_sizes + [1], activation)

    def forward(self, obs, act):
        # state input is (L, Hin), price input is (Hprice, )
        state_feature, (hn, cn) = self.state_net(obs["net"]) 
        price_feature = self.price_net(obs["price"])
        # Concatenate the last hidden state of LSTM and price feature
        q = self.q(torch.cat([hn[-1], price_feature, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ActorMLP(ActorBase):
    def __init__(self, state_dim:int, price_dim:int, act_dim:int, hidden_sizes:list[int], activation:Any, act_limit:tuple[float, float]):
        super().__init__(state_dim, price_dim, act_dim, hidden_sizes, activation, act_limit)
        self.net = mlp([state_dim + price_dim] + hidden_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # Take the last element along the second to last dimension of obs["net"], keeping other dimensions unchanged
        obs_net_last = obs["net"].select(dim=-2, index=-1)
        net_out = self.net(torch.cat((obs_net_last, obs["price"]), dim=-1))  # Concatenate last state and price
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit[0] + 0.5 * (pi_action + 1.0) * (self.act_limit[1] - self.act_limit[0])

        return pi_action, logp_pi


class QFunctionMLP(QFunctionBase):
    def __init__(self, state_dim:int, price_dim:int, act_dim:int, hidden_sizes:list[int], activation:Any):
        super().__init__(state_dim, price_dim, act_dim, hidden_sizes, activation)  # price_dim is not used in MLPQFunction
        self.q = mlp([state_dim + price_dim + act_dim] + hidden_sizes + [1], activation)

    def forward(self, obs, act):
        obs_net_last = obs["net"].select(dim=-2, index=-1)
        q = self.q(torch.cat([obs_net_last, obs["price"], act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ActorCritic(nn.Module):
    def __init__(self, observation_space:gym.spaces.Space, action_space:gym.spaces.Space, hidden_sizes=(256,256),
                 activation=nn.ReLU, actor:type = ActorMLP, q_function:type = QFunctionMLP):
        super().__init__()

        assert isinstance(observation_space, gym.spaces.Dict), "Observation space must be a Dict space."
        net_space = observation_space["net"]
        assert (
            isinstance(net_space, gym.spaces.Sequence) and 
            isinstance(net_space.feature_space, gym.spaces.Box)
        ), "Net observation space must be a Sequence space."
        state_dim = net_space.feature_space.shape[0]

        price_space = observation_space["price"]
        assert isinstance(price_space, gym.spaces.Box), "Price observation space must be a Box space."
        price_dim = price_space.shape[0]

        assert isinstance(action_space, gym.spaces.Box), "Action space must be a Box space."
        act_dim = action_space.shape[0]
        act_limit = (action_space.low[0], action_space.high[0])

        # build policy and value functions
        self.pi:ActorBase = actor(state_dim, price_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1:QFunctionBase = q_function(state_dim, price_dim, act_dim, hidden_sizes, activation)
        self.q2:QFunctionBase = q_function(state_dim, price_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        obs["net"] = torch.as_tensor(obs["net"], dtype=torch.float32)
        obs["price"] = torch.as_tensor(obs["price"], dtype=torch.float32)
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            a:torch.Tensor
            return a.numpy()
