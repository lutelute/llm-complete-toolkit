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

#### 📦 自動インストール（推奨）

**Linux/macOS:**
```bash
# リポジトリのクローン
git clone https://github.com/lutelute/llm-complete-toolkit.git
cd llm-complete-toolkit

# 自動インストール実行
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
# リポジトリのクローン
git clone https://github.com/lutelute/llm-complete-toolkit.git
cd llm-complete-toolkit

# 自動インストール実行
install.bat
```

#### 🔧 手動インストール

```bash
# リポジトリのクローン
git clone https://github.com/lutelute/llm-complete-toolkit.git
cd llm-complete-toolkit

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# または
venv\Scripts\activate     # Windows

# 依存関係のインストール
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# セットアップスクリプト実行
python setup.py

# GPU最適化（オプション）
pip install flash-attn --no-build-isolation
```

#### 🐳 Docker利用

```bash
# Dockerイメージのビルド
docker build -t llm-complete-toolkit .

# コンテナの実行
docker run -it --gpus all -v $(pwd):/workspace llm-complete-toolkit
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

### 4. 学習用データの準備

```bash
# 学習用データフォルダにファイルを配置（全形式混在OK）
cp /path/to/your/mixed/files/* training_data/raw/

# 学習用データの抽出・変換
python main.py extract-training-data --split-data --instruction-format

# 配置されたファイルを確認
ls -la training_data/raw/
```

### 5. ドキュメント処理（従来の方法）

```bash
# PDFやMarkdownファイルからデータを抽出
python main.py extract data/ outputs/ --format jsonl --instruction-format
```

### 6. モデルトレーニング

```bash
# 学習用データで訓練
python main.py train-lora --train-data training_data/datasets/training/train.jsonl --eval-data training_data/datasets/validation/eval.jsonl

# QLoRAファインチューニング（メモリ効率的）
python main.py train-qlora --train-data training_data/datasets/training/train.jsonl

# PPO強化学習
python main.py train-rl --algorithm ppo

# DQN強化学習
python main.py train-rl --algorithm dqn
```

## 📚 詳細な使用方法

### 学習用データ管理

#### サポートファイル形式
- **PDF**: .pdf
- **Markdown**: .md, .markdown
- **JSON**: .json, .jsonl
- **テキスト**: .txt

#### 処理例
```bash
# 学習用データの抽出（全形式自動検出）
python main.py extract-training-data

# データ分割付き処理（推奨）
python main.py extract-training-data --split-data --instruction-format

# 特定形式のみ処理
python main.py extract-training-data --format pdf
python main.py extract-training-data --format json

# カスタム入力フォルダを指定
python main.py extract-training-data --input-dir /path/to/your/documents

# 詳細オプション
python main.py extract-training-data \
  --input-dir /path/to/your/documents \
  --output-dir training_data/processed \
  --output-format jsonl \
  --chunk-size 1024 \
  --instruction-format \
  --split-data \
  --train-ratio 0.8 \
  --val-ratio 0.1
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

### 📋 システム要件チェック

#### Python環境の確認
```bash
# Python バージョン確認
python --version  # 3.8以上必要

# 仮想環境の確認
which python  # venv内のpythonを使用しているか確認
```

#### GPU環境の確認
```bash
# CUDA環境の確認
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Apple Silicon (MPS) の確認
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### 🔧 よくある問題と解決方法

#### 1. CUDA out of memory
```bash
# バッチサイズを小さくする
python main.py train-lora --train-data data/train.jsonl --batch-size 1

# QLoRAを使用する（メモリ効率的）
python main.py train-qlora --train-data data/train.jsonl --batch-size 1

# 勾配累積を使用する
python main.py train-lora --train-data data/train.jsonl --batch-size 1 --gradient-accumulation-steps 8
```

#### 2. モデル読み込みエラー
```bash
# HuggingFaceキャッシュをクリア
rm -rf ~/.cache/huggingface/

# 依存関係を更新
pip install --upgrade transformers torch accelerate

# 特定のモデルを再ダウンロード
python scripts/download_models.py --model microsoft/DialoGPT-medium
```

#### 3. インストールエラー
```bash
# pip を最新版にアップデート
pip install --upgrade pip setuptools wheel

# 依存関係を個別インストール
pip install torch transformers datasets

# requirements.txtを段階的にインストール
pip install -r requirements.txt --no-deps
pip install -r requirements.txt
```

#### 4. Flash Attention インストール失敗
```bash
# CUDA環境でのインストール
pip install flash-attn --no-build-isolation

# 環境が合わない場合は無効化
# requirements.txtでコメントアウト
```

#### 5. パフォーマンスの問題
```bash
# CPU使用数を制限
export OMP_NUM_THREADS=4

# メモリ使用量を制限
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 混合精度を有効化
python main.py train-lora --train-data data/train.jsonl --fp16
```

### 🏥 診断スクリプト

```bash
# 環境診断の実行
python -c "
import torch
import transformers
import sys

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('GPU Count:', torch.cuda.device_count())
    print('GPU Name:', torch.cuda.get_device_name(0))
"
```

### 📞 サポート

問題が解決しない場合は、以下の情報と共にIssueを作成してください：

1. **環境情報**: OS、Python版、GPU情報
2. **エラーメッセージ**: 完全なエラーログ
3. **実行コマンド**: 実際に実行したコマンド
4. **設定ファイル**: 使用したconfig.yamlの内容

```bash
# 環境情報を取得
python setup.py --debug > debug_info.txt
```

### 🔗 参考リンク

- [PyTorch GPU サポート](https://pytorch.org/get-started/locally/)
- [HuggingFace Transformers ドキュメント](https://huggingface.co/docs/transformers)
- [PEFT (Parameter Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [Stable Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)

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