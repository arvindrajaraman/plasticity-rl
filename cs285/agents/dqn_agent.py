from typing import Sequence, Callable, Tuple, Optional
import time
import os

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from cs285.infrastructure.utils import *

import numpy as np

import matplotlib.pyplot as plt

import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure.pytorch_util import DeepFFNN

class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
        weight_plot_freq: int = 100,
        logdir: str = None,
        regularizer: str = 'none',
        lambda_: float = 0.01,
        layer_discount: float = 1.0,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions) ## TODO: replace with custom critic.
        self.critic_weights_t0 = get_weights_by_layer(self.critic)

        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)
        self.device = ptu.device

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.critic_iter = 0
        self.weight_plot_freq = weight_plot_freq
        self.logdir = logdir
        self.regularizer = regularizer
        self.lambda_ = lambda_
        self.layer_discount = layer_discount

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        observation = observation.to(device=self.device)

        # TODO(student): get the action from the critic using an epsilon-greedy strategy    
        qa_values = self.critic(observation)
        max_idx = torch.argmax(qa_values, dim=-1)
        dist_array = [epsilon / (self.num_actions - 1)] * self.num_actions
        dist_array[max_idx] = 1 - epsilon

        dist = Categorical(torch.tensor(dist_array))
        action = dist.sample()

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                next_action = torch.argmax(self.critic(next_obs), dim=1).unsqueeze(1)
            else:
                next_action = torch.argmax(next_qa_values, dim=1).unsqueeze(1)
            
            next_q_values = torch.gather(next_qa_values, 1, next_action).squeeze(1)
            target_values = reward + (self.discount * next_q_values * (1.0 - done.float()))

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)  # Compute from the data actions; see torch.gather
        
        ## Plasticity-related regularizers ##
        loss = self.critic_loss(q_values, target_values) 
        ## loss = critic_loss + l2_reg + regenerative_reg + singular_loss 
        if self.regularizer == 'none':
            pass
        elif self.regularizer == 'weight_mag':
            for i, (name, param) in enumerate(self.critic.named_parameters()):
                loss += self.lambda_ * (self.layer_discount ** i) * torch.norm(param, p=2)
        elif self.regularizer == 'regenerative_reg':
            for i, (name, param) in enumerate(self.critic.named_parameters()):
                loss += self.lambda_ * (self.layer_discount ** i) * torch.norm(param - self.critic_weights_t0[i], p=2)
        else:
            raise ValueError(f'Unknown regularizer: {self.regularizer}')

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        ## Plasticity-related code ##

        # Concatenate weights for easy analysis
        flat_weights = get_weights(self.critic)
        flat_weights_by_layer = get_weights_by_layer(self.critic)

        # Plot weight distribution
        if self.critic_iter % self.weight_plot_freq == 0:
            plt.figure(figsize=(3, 3))
            plt.hist(flat_weights, bins=100, color='green', range=(-3, 3), density=True)
            plt.text(0.5, 0.5, f'iter={self.critic_iter}')
            dir_prefix = 'data/' + self.logdir + f'/critic_weight_dist/'
            if not (os.path.exists(dir_prefix)):
                os.makedirs(dir_prefix)
            plt.savefig(dir_prefix + f'dist_{self.critic_iter}.png')

        info = {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

        # Calculate weight magnitude
        weight_magnitude = np.linalg.norm(flat_weights)
        info["critic_weight_mag"] = weight_magnitude.item()

        for i, weights in enumerate(flat_weights_by_layer):
            layer_weight_magnitude = np.linalg.norm(weights)
            info["critic_weight_mag_layer_{}".format(i)] = layer_weight_magnitude.item()

        # Calculate effective rank
        # Use utils compute_effective_rank and predict function
        # n_layers = len(flat_weights_by_layer)
        # matrix = self.critic(obs)[1]
        
        # for layer_idx in range(n_layers):
            # rank = compute_effective_rank(matrix[layer_idx])
            # info["critic_effective_rank_layer_{}".format(layer_idx)] = rank

        self.critic_iter += 1
        return info


    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs.to(device=self.device), action.to(device = self.device), reward.to(device= self.device), \
                                          next_obs.to(device= self.device), done.to(device = self.device))

        if step % self.target_update_period == 0:
            self.update_target_critic()
        
        return critic_stats
