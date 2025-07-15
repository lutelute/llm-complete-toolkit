# 🚀 LLM Complete Toolkit

PDFやMarkdownファイルからのドキュメント処理と、強化学習・転移学習を用いたLLMトレーニングの統合ツールキット

## ✨ 主な特徴

### 📄 ドキュメント処理
- **PDF解析**: PDFplumber + PyPDF2によるテキスト抽出
- **Markdown解析**: フロントマター対応、セクション分割
- **LM Studio変換**: JSONL形式での学習データ生成
- **自動チャンク分割**: 最適なサイズでのテキスト分割

### 🧠 転移学習
- **LoRA**: 効率的なファインチューニング
- **QLoRA**: 量子化による超メモリ効率的トレーニング
- **自動データセット準備**: インストラクション形式対応
- **HuggingFace統合**: 最新のTransformersライブラリ対応

### 🎯 強化学習
- **PPO**: 安定したポリシー勾配手法
- **DQN**: 価値ベース学習（Dueling DQN対応）
- **カスタム環境**: テキスト処理環境のサンプル実装
- **TensorBoard統合**: リアルタイム学習監視

### 🛠️ 統合機能
- **統一ランチャー**: 全機能を単一コマンドで実行
- **YAML設定**: 柔軟な設定管理
- **メトリクス記録**: TensorBoard/W&B対応
- **チェックポイント管理**: 自動保存・復元

## 📁 プロジェクト構造

```
llm-complete-toolkit/
├── main.py                      # 統合メインランチャー
├── configs/
│   └── config.yaml             # 統合設定ファイル
├── document_processing/         # ドキュメント処理モジュール
│   ├── parsers/
│   │   ├── pdf_parser.py       # PDF解析
│   │   └── markdown_parser.py  # Markdown解析
│   └── converters/
│       └── lm_studio_converter.py  # LM Studio変換
├── training_methods/            # 学習手法モジュール
│   ├── reinforcement_learning/
│   │   └── agents/
│   │       ├── ppo_agent.py    # PPOエージェント
│   │       └── dqn_agent.py    # DQNエージェント
│   └── transfer_learning/
│       └── models/
│           ├── lora_model.py   # LoRAモデル
│           └── qlora_model.py  # QLoRAモデル
├── shared_utils/               # 共通ユーティリティ
│   ├── data_loader.py         # データローダー
│   ├── training_utils.py      # トレーニングユーティリティ
│   └── file_utils.py          # ファイルユーティリティ
├── scripts/                   # 個別トレーニングスクリプト
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── train_rl.py
│   └── download_models.py      # Hugging Faceモデルダウンロード
├── models/                    # モデル保存ディレクトリ
│   ├── base_models/           # 未学習（事前学習済み）モデル
│   ├── fine_tuned_models/     # ファインチューニング済みモデル
│   └── trained_models/        # 強化学習トレーニング済みモデル
├── data/                      # データディレクトリ
├── outputs/                   # 出力ディレクトリ
└── requirements.txt          # 依存関係
```

## 🚀 クイックスタート

### 1. インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd llm-complete-toolkit

# 依存関係のインストール
pip install -r requirements.txt

# Flash Attention (オプション、GPUでの高速化)
pip install flash-attn --no-build-isolation
```

### 2. モデルのダウンロード

```bash
# 人気モデルの一覧表示
python scripts/download_models.py --list-popular

# 単一モデルのダウンロード
python scripts/download_models.py --model microsoft/DialoGPT-medium

# 複数モデルのダウンロード
python scripts/download_models.py --model gpt2 --model microsoft/DialoGPT-small
```

### 3. サンプルデータの作成

```bash
python main.py create-samples
```

### 4. ドキュメント処理

```bash
# PDFやMarkdownファイルからデータを抽出
python main.py extract data/ outputs/ --format jsonl --instruction-format
```

### 5. モデルトレーニング

```bash
# LoRAファインチューニング
python main.py train-lora --train-data data/train.jsonl

# QLoRAファインチューニング（メモリ効率的）
python main.py train-qlora --train-data data/train.jsonl

# PPO強化学習
python main.py train-rl --algorithm ppo

# DQN強化学習
python main.py train-rl --algorithm dqn
```

## 📚 詳細な使用方法

### ドキュメント処理

#### サポートファイル形式
- **PDF**: .pdf
- **Markdown**: .md, .markdown
- **テキスト**: .txt

#### 処理例
```bash
# 基本的な抽出
python main.py extract documents/ outputs/

# 詳細オプション
python main.py extract documents/ outputs/ \
  --format jsonl \
  --chunk-size 1024 \
  --instruction-format
```

### 転移学習

#### LoRAファインチューニング
```bash
python main.py train-lora \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --model-name microsoft/DialoGPT-medium \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4
```

#### QLoRAファインチューニング
```bash
python main.py train-qlora \
  --train-data data/train.jsonl \
  --model-name microsoft/DialoGPT-medium \
  --batch-size 1 \
  --epochs 3
```

### 強化学習

#### PPO学習
```bash
python main.py train-rl \
  --algorithm ppo \
  --episodes 1000 \
  --config configs/config.yaml
```

#### DQN学習
```bash
python main.py train-rl \
  --algorithm dqn \
  --episodes 1000
```

## ⚙️ 設定ファイル

`configs/config.yaml`で全てのパラメータをカスタマイズできます：

```yaml
# 共通設定
common:
  device: "cuda"
  seed: 42
  log_level: "INFO"

# 転移学習設定
transfer_learning:
  lora:
    model_name: "microsoft/DialoGPT-medium"
    lora_r: 16
    lora_alpha: 32
    training:
      num_train_epochs: 3
      learning_rate: 2e-4

# 強化学習設定
reinforcement_learning:
  ppo:
    learning_rate: 3e-4
    gamma: 0.99
    training:
      max_episodes: 1000
```

## 📊 データ形式

### インストラクションデータ（JSONL）
```json
{
  "instruction": "Pythonでリストを作成する方法を教えてください。",
  "input": "",
  "output": "Pythonでリストを作成するには、角括弧[]を使用します。例: my_list = [1, 2, 3]"
}
```

### LM Studio変換データ
```json
{
  "text": "抽出されたテキスト内容",
  "source": "ファイルパス",
  "title": "ドキュメントタイトル",
  "type": "pdf",
  "chunk_id": 1,
  "total_chunks": 5,
  "metadata": {...}
}
```

## 🎯 出力ファイル

### ドキュメント処理
- `training_data.jsonl`: LM Studio用学習データ
- `instruction_data.jsonl`: インストラクション形式データ

### LoRA/QLoRA
- `pytorch_model.bin`: LoRA重み
- `adapter_config.json`: アダプター設定
- `training_config.yaml`: トレーニング設定

### 強化学習
- `{algorithm}_final.pt`: 最終モデル
- `{algorithm}_episode_{n}.pt`: 定期保存モデル
- `training_curves.png`: トレーニング曲線

## 📈 モニタリング

### TensorBoard
```bash
tensorboard --logdir=outputs/logs/tensorboard
```

### Weights & Biases
設定ファイルでW&Bを有効化：
```yaml
logging:
  use_wandb: true
  wandb_project: "my-llm-project"
```

## 🔧 カスタマイズ

### 新しいパーサーの追加
```python
# document_processing/parsers/my_parser.py
class MyParser:
    def parse(self, file_path):
        # カスタム解析ロジック
        return documents
```

### カスタム強化学習環境
```python
# training_methods/reinforcement_learning/environments/
class MyEnvironment:
    def reset(self):
        return initial_state
    
    def step(self, action):
        return next_state, reward, done, info
```

## 💡 使用例とワークフロー

### 1. ドキュメントからLLMファインチューニングまで
```bash
# ステップ1: サンプルデータ作成
python main.py create-samples

# ステップ2: ドキュメント処理
python main.py extract data/ outputs/ --instruction-format

# ステップ3: LoRAファインチューニング
python main.py train-lora --train-data outputs/instruction_data.jsonl
```

### 2. 大規模モデルの効率的トレーニング
```bash
# QLoRAで大規模モデルを効率的にファインチューニング
python main.py train-qlora \
  --train-data data/train.jsonl \
  --model-name microsoft/DialoGPT-large \
  --batch-size 1 \
  --epochs 2
```

### 3. 強化学習でのポリシー最適化
```bash
# PPOで安定したポリシー学習
python main.py train-rl --algorithm ppo --episodes 2000
```

## ❗ トラブルシューティング

### よくある問題

#### CUDA out of memory
```bash
# バッチサイズを小さくする
python main.py train-qlora --train-data data/train.jsonl --batch-size 1

# QLoRAを使用する
python main.py train-qlora --train-data data/train.jsonl
```

#### モデル読み込みエラー
```bash
# キャッシュをクリア
rm -rf ~/.cache/huggingface/

# 依存関係を更新
pip install --upgrade transformers torch
```

#### メモリ不足
- QLoRAを使用
- gradient_accumulation_stepsを増やす
- per_device_train_batch_sizeを減らす

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🆘 サポート

問題や質問がある場合は、GitHubのIssuesページで報告してください。

---

**LLM Complete Toolkit** - ドキュメント処理から機械学習まで、一貫したワークフローを提供する統合ツールキット 🚀