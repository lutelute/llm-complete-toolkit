# Trained Models Directory

このディレクトリには、強化学習（PPO/DQN）でトレーニングされたモデルが保存されます。

## 使用方法

### PPO強化学習
```bash
# PPO強化学習実行
python main.py train-rl --algorithm ppo --output-dir models/trained_models/ppo_model

# エピソード数を指定
python main.py train-rl --algorithm ppo --episodes 2000
```

### DQN強化学習
```bash
# DQN強化学習実行
python main.py train-rl --algorithm dqn --output-dir models/trained_models/dqn_model

# 設定ファイルを使用
python main.py train-rl --algorithm dqn --config configs/config.yaml
```

## ディレクトリ構造

各強化学習モデルは以下の形式で保存されます：

### PPOモデル
```
trained_models/
├── ppo_model_20250115_143022/
│   ├── ppo_final.pt                 # 最終モデル
│   ├── ppo_best.pt                  # 最高性能モデル
│   ├── training_config.yaml         # トレーニング設定
│   ├── training_log.txt             # トレーニングログ
│   ├── rewards_history.json         # 報酬履歴
│   ├── training_curves.png          # 学習曲線
│   └── checkpoints/                 # エピソード別チェックポイント
│       ├── ppo_episode_100.pt
│       ├── ppo_episode_200.pt
│       └── ...
```

### DQNモデル
```
trained_models/
├── dqn_model_20250115_143022/
│   ├── dqn_final.pt                 # 最終モデル
│   ├── dqn_best.pt                  # 最高性能モデル
│   ├── target_network.pt            # ターゲットネットワーク
│   ├── training_config.yaml         # トレーニング設定
│   ├── training_log.txt             # トレーニングログ
│   ├── rewards_history.json         # 報酬履歴
│   ├── training_curves.png          # 学習曲線
│   └── checkpoints/                 # エピソード別チェックポイント
│       ├── dqn_episode_100.pt
│       ├── dqn_episode_200.pt
│       └── ...
```

## モデルの使用方法

### PPOモデルの読み込み
```python
import torch
from training_methods.reinforcement_learning.agents.ppo_agent import PPOAgent

# モデルの読み込み
model_path = "models/trained_models/ppo_model_20250115_143022/ppo_final.pt"
agent = PPOAgent(state_dim=128, action_dim=10, hidden_dim=256)
agent.load_model(model_path)

# 推論
state = torch.randn(1, 128)
action = agent.select_action(state)
```

### DQNモデルの読み込み
```python
import torch
from training_methods.reinforcement_learning.agents.dqn_agent import DQNAgent

# モデルの読み込み
model_path = "models/trained_models/dqn_model_20250115_143022/dqn_final.pt"
agent = DQNAgent(state_dim=128, action_dim=10, hidden_dim=256)
agent.load_model(model_path)

# 推論
state = torch.randn(1, 128)
action = agent.select_action(state, epsilon=0.0)  # 探索なし
```

## 学習アルゴリズム

### PPO (Proximal Policy Optimization)
- **特徴**: 安定したポリシー勾配手法
- **適用場面**: 連続行動空間、安定性重視
- **主要パラメータ**:
  - `learning_rate`: 学習率 (デフォルト: 3e-4)
  - `gamma`: 割引率 (デフォルト: 0.99)
  - `eps_clip`: クリッピング範囲 (デフォルト: 0.2)
  - `k_epochs`: 更新回数 (デフォルト: 4)

### DQN (Deep Q-Network)
- **特徴**: 価値ベース学習、Dueling DQN対応
- **適用場面**: 離散行動空間、価値関数学習
- **主要パラメータ**:
  - `learning_rate`: 学習率 (デフォルト: 1e-3)
  - `gamma`: 割引率 (デフォルト: 0.99)
  - `epsilon_start`: 初期探索率 (デフォルト: 1.0)
  - `epsilon_end`: 最終探索率 (デフォルト: 0.01)
  - `buffer_size`: 経験バッファサイズ (デフォルト: 10000)

## 環境設定

### カスタム環境の作成
```python
import gymnasium as gym
from gymnasium import spaces

class CustomTextEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(128,))
    
    def reset(self, seed=None, options=None):
        # 初期状態の設定
        return self.observation_space.sample(), {}
    
    def step(self, action):
        # 行動の実行
        next_state = self.observation_space.sample()
        reward = self.calculate_reward(action)
        done = self.is_episode_done()
        info = {}
        return next_state, reward, done, False, info
    
    def calculate_reward(self, action):
        # 報酬の計算
        return 1.0 if action == 0 else 0.0
    
    def is_episode_done(self):
        # エピソード終了条件
        return False
```

## 評価メトリクス

### 学習中のメトリクス
- **平均報酬**: エピソードごとの平均報酬
- **エピソード長**: 各エピソードのステップ数
- **損失**: 価値関数やポリシーの損失
- **探索率**: ε-グリーディの探索率（DQNの場合）

### 保存されるファイル
- `rewards_history.json`: 報酬の履歴データ
- `training_curves.png`: 学習曲線の可視化
- `training_log.txt`: 詳細な学習ログ

## パフォーマンス比較

| アルゴリズム | 収束性 | 安定性 | 計算効率 | 適用範囲 |
|-------------|--------|--------|----------|----------|
| PPO | 高 | 高 | 中 | 連続・離散両方 |
| DQN | 中 | 中 | 高 | 離散行動のみ |

## 注意事項

- 学習には時間がかかる場合があります（GPUの使用を推奨）
- 環境によって最適なハイパーパラメータが異なります
- 定期的にチェックポイントを保存することで学習の中断・再開が可能です
- 評価は複数回実行して平均をとることを推奨します