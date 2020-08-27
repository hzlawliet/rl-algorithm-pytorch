#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/12/19 11:12 上午

@author: wanghengzhi
"""
import torch
from torch import nn
from torch.distributions import Categorical


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, n_latent_var=None):
        super(ActorCritic, self).__init__()
        self.device = device
        # actor
        if n_latent_var is None or len(n_latent_var) != 2:
            n_latent_var = [128, 32]
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var[0]),
            nn.Tanh(),
            nn.Linear(n_latent_var[0], n_latent_var[1]),
            nn.Tanh(),
            nn.Linear(n_latent_var[1], action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var[0]),
            nn.BatchNorm1d(n_latent_var[0], affine=True),
            nn.Tanh(),
            nn.Linear(n_latent_var[0], n_latent_var[1]),
            nn.BatchNorm1d(n_latent_var[1], affine=True),
            nn.Tanh(),
            nn.Linear(n_latent_var[1], 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, action, reward, done, memory):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = torch.LongTensor([action]).reshape(torch.Size()).to(self.device)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(state_dim, action_dim, self.device, n_latent_var).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, self.device, n_latent_var).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        loss_list = []
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            ppo_loss = loss.mean()
            ppo_loss.backward()
            self.optimizer.step()
            loss_list.append(ppo_loss.item())

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss_list

