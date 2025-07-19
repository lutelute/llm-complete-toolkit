#!/usr/bin/env python3
"""
強化学習トレーニングスクリプト
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_methods.reinforcement_learning.agents.ppo_agent import PPOAgent
from training_methods.reinforcement_learning.agents.dqn_agent import DQNAgent
from shared_utils.training_utils import setup_logging, set_seed, MetricsLogger, TrainingMetrics


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class SimpleTextEnvironment:
    """シンプルなテキスト処理環境（デモ用）"""
    
    def __init__(self, vocab_size: int = 1000, max_length: int = 50):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.reset()
    
    def reset(self):
        """環境をリセット"""
        self.current_text = np.random.randint(0, self.vocab_size, size=(self.max_length,))
        self.step_count = 0
        self.target_text = np.random.randint(0, self.vocab_size, size=(self.max_length,))
        return self.get_state()
    
    def get_state(self):
        """現在の状態を取得"""
        # テキスト状態を簡略化した特徴ベクトルに変換
        state = np.concatenate([
            self.current_text / self.vocab_size,  # 正規化
            self.target_text / self.vocab_size,   # 正規化
            [self.step_count / self.max_length]   # ステップ進捗
        ])
        return state.astype(np.float32)
    
    def step(self, action):
        """アクションを実行"""
        # アクション: 0=変更なし, 1-9=特定の変更操作
        if action > 0 and self.step_count < self.max_length:
            # テキストの一部を変更
            pos = self.step_count % self.max_length
            self.current_text[pos] = (self.current_text[pos] + action) % self.vocab_size
        
        self.step_count += 1
        
        # 報酬計算（目標テキストとの類似度）
        similarity = np.mean(self.current_text == self.target_text)
        reward = similarity - 0.5  # -0.5 to 0.5 range
        
        # 終了条件
        done = self.step_count >= self.max_length or similarity > 0.9
        
        return self.get_state(), reward, done, {}


def train_ppo(config: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """PPOエージェントの訓練"""
    ppo_config = config['reinforcement_learning']['ppo']
    
    # 環境の初期化
    env = SimpleTextEnvironment()
    state_dim = len(env.get_state())
    action_dim = ppo_config['action_dim']
    
    # エージェントの初期化
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=ppo_config['learning_rate'],
        gamma=ppo_config['gamma'],
        eps_clip=ppo_config['eps_clip'],
        k_epochs=ppo_config['k_epochs']
    )
    
    # メトリクスロガー
    metrics_logger = MetricsLogger(
        log_dir=str(output_dir / "logs"),
        use_tensorboard=config['logging']['use_tensorboard'],
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging']['wandb_project']
    )
    
    # トレーニングループ
    training_config = ppo_config['training']
    max_episodes = training_config['max_episodes']
    max_steps = training_config['max_steps_per_episode']
    update_frequency = training_config['update_frequency']
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # アクション選択
            action, log_prob, value = agent.select_action(state)
            
            # 環境でアクションを実行
            next_state, reward, done, _ = env.step(action)
            
            # 経験を保存
            agent.store_transition(state, action, reward, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 定期的にモデルを更新
        if total_steps % update_frequency == 0:
            losses = agent.update()
            
            # メトリクス記録
            if losses:
                metrics = TrainingMetrics(
                    epoch=episode,
                    step=total_steps,
                    loss=losses['total_loss'],
                    learning_rate=ppo_config['learning_rate'],
                    timestamp=time.time(),
                    additional_metrics={
                        'episode_reward': episode_reward,
                        'episode_steps': episode_steps,
                        'actor_loss': losses['actor_loss'],
                        'critic_loss': losses['critic_loss']
                    }
                )
                metrics_logger.log_metrics(metrics)
        
        # 進捗表示
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.3f}")
        
        # モデル保存
        if episode % training_config['save_frequency'] == 0:
            save_path = output_dir / f"ppo_episode_{episode}.pt"
            agent.save(str(save_path))
    
    # 最終モデル保存
    final_path = output_dir / "ppo_final.pt"
    agent.save(str(final_path))
    
    # トレーニング曲線をプロット
    metrics_logger.plot_training_curves(str(output_dir / "ppo_training_curves.png"))
    metrics_logger.close()
    
    logger.info(f"PPOトレーニング完了: {output_dir}")


def train_dqn(config: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """DQNエージェントの訓練"""
    dqn_config = config['reinforcement_learning']['dqn']
    
    # 環境の初期化
    env = SimpleTextEnvironment()
    state_dim = len(env.get_state())
    action_dim = dqn_config['action_dim']
    
    # エージェントの初期化
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=dqn_config['learning_rate'],
        gamma=dqn_config['gamma'],
        epsilon_start=dqn_config['epsilon_start'],
        epsilon_end=dqn_config['epsilon_end'],
        epsilon_decay=dqn_config['epsilon_decay'],
        buffer_size=dqn_config['buffer_size'],
        batch_size=dqn_config['batch_size'],
        target_update=dqn_config['target_update'],
        use_dueling=dqn_config['use_dueling']
    )
    
    # メトリクスロガー
    metrics_logger = MetricsLogger(
        log_dir=str(output_dir / "logs"),
        use_tensorboard=config['logging']['use_tensorboard'],
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging']['wandb_project']
    )
    
    # トレーニングループ
    training_config = dqn_config['training']
    max_episodes = training_config['max_episodes']
    max_steps = training_config['max_steps_per_episode']
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # アクション選択
            action = agent.select_action(state, training=True)
            
            # 環境でアクションを実行
            next_state, reward, done, _ = env.step(action)
            
            # 経験を保存
            agent.store_transition(state, action, reward, next_state, done)
            
            # モデル更新
            losses = agent.update()
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # メトリクス記録
            if losses and total_steps % 10 == 0:
                metrics = TrainingMetrics(
                    epoch=episode,
                    step=total_steps,
                    loss=losses['loss'],
                    learning_rate=dqn_config['learning_rate'],
                    timestamp=time.time(),
                    additional_metrics={
                        'episode_reward': episode_reward,
                        'epsilon': losses['epsilon'],
                        'q_value_mean': losses['q_value_mean']
                    }
                )
                metrics_logger.log_metrics(metrics)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 進捗表示
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
        
        # モデル保存
        if episode % training_config['save_frequency'] == 0:
            save_path = output_dir / f"dqn_episode_{episode}.pt"
            agent.save(str(save_path))
    
    # 最終モデル保存
    final_path = output_dir / "dqn_final.pt"
    agent.save(str(final_path))
    
    # トレーニング曲線をプロット
    metrics_logger.plot_training_curves(str(output_dir / "dqn_training_curves.png"))
    metrics_logger.close()
    
    logger.info(f"DQNトレーニング完了: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='強化学習トレーニング')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn'], required=True,
                       help='使用するアルゴリズム')
    parser.add_argument('--output-dir', type=str, default='./outputs/rl',
                       help='出力ディレクトリ')
    parser.add_argument('--episodes', type=int,
                       help='エピソード数（設定ファイルを上書き）')
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細ログ表示')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config(args.config)
    common_config = config['common']
    
    # ログ設定
    log_level = "DEBUG" if args.verbose else common_config.get('log_level', 'INFO')
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # シード設定
    set_seed(args.seed)
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir) / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"強化学習トレーニング開始: {args.algorithm.upper()}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    try:
        if args.algorithm == 'ppo':
            train_ppo(config, output_dir, logger)
        elif args.algorithm == 'dqn':
            train_dqn(config, output_dir, logger)
        
        # 設定ファイルの保存
        config_save_path = output_dir / "training_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()