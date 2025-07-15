"""
PPO (Proximal Policy Optimization) エージェントの実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def get_batch(self):
        return (
            torch.stack(self.states),
            torch.tensor(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.bool),
            torch.stack(self.log_probs),
            torch.stack(self.values)
        )


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # 共通の特徴抽出層
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actorネットワーク（ポリシー）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Criticネットワーク（価値関数）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def act(self, state):
        with torch.no_grad():
            action_probs, state_value = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, state_value


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()
        
        self.logger = logging.getLogger(__name__)
    
    def select_action(self, state):
        """状態に対してアクションを選択"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.policy.act(state)
        
        return action, log_prob, value
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """経験をメモリに保存"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.memory.store(state_tensor, action, reward, done, log_prob, value)
    
    def update(self):
        """PPOアルゴリズムによるポリシー更新"""
        states, actions, rewards, dones, old_log_probs, old_values = self.memory.get_batch()
        
        # 割引報酬の計算
        discounted_rewards = self._compute_discounted_rewards(rewards, dones)
        
        # アドバンテージの計算
        advantages = discounted_rewards - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # K回のエポックで更新
        for _ in range(self.k_epochs):
            # 現在のポリシーでの評価
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # 確率比率の計算
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPOの目的関数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Criticの損失（価値関数の損失）
            critic_loss = nn.MSELoss()(state_values.squeeze(), discounted_rewards)
            
            # エントロピーボーナス
            entropy_loss = -entropy.mean()
            
            # 総損失
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # 勾配更新
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # メモリをクリア
        self.memory.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_discounted_rewards(self, rewards, dones):
        """割引報酬の計算"""
        discounted_rewards = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        
        return torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
    
    def save(self, path: str):
        """モデルの保存"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        self.logger.info(f"モデルを保存しました: {path}")
    
    def load(self, path: str):
        """モデルの読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"モデルを読み込みました: {path}")