# 🚀 LLM Complete Toolkit クイックスタートガイド

## 📋 概要
このガイドでは、LLM Complete Toolkitの基本的な使い方を5つのステップで説明します。

## 🎯 前提条件
- Python 3.8以上
- 8GB以上のRAM（推奨：16GB以上）
- GPU（オプション、CUDA対応またはMPS対応）

## 🔧 ステップ1: インストール

### 自動インストール（推奨）
```bash
# リポジトリのクローン
git clone https://github.com/lutelute/llm-complete-toolkit.git
cd llm-complete-toolkit

# 自動インストール実行
./install.sh  # Linux/macOS
# または
install.bat   # Windows
```

### 手動インストール
```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 依存関係のインストール
pip install -r requirements.txt
python setup.py
```

## 📥 ステップ2: モデルのダウンロード

```bash
# 利用可能なモデル一覧を確認
python scripts/download_models.py --list-popular

# 軽量モデルをダウンロード（初回推奨）
python scripts/download_models.py --model microsoft/DialoGPT-small

# 複数モデルをダウンロード
python scripts/download_models.py --model gpt2 --model microsoft/DialoGPT-medium
```

## 📄 ステップ3: サンプルデータの準備

```bash
# サンプルデータの生成
python main.py create-samples

# 生成されたファイルを確認
ls data/
# sample_document.md  train.jsonl  eval.jsonl
```

## 📊 ステップ4: ドキュメント処理

```bash
# PDFやMarkdownファイルからデータを抽出
python main.py extract data/ outputs/ --format jsonl --instruction-format

# 出力ファイルを確認
ls outputs/
# training_data.jsonl  instruction_data.jsonl
```

## 🤖 ステップ5: モデルトレーニング

### A. LoRAファインチューニング
```bash
# LoRAファインチューニング実行
python main.py train-lora --train-data data/train.jsonl --eval-data data/eval.jsonl

# 高速化オプション
python main.py train-lora --train-data data/train.jsonl --batch-size 2 --epochs 1
```

### B. QLoRAファインチューニング（メモリ効率）
```bash
# QLoRAファインチューニング実行
python main.py train-qlora --train-data data/train.jsonl --batch-size 1

# 大規模モデルでのトレーニング
python main.py train-qlora --train-data data/train.jsonl --model-name microsoft/DialoGPT-large
```

### C. 強化学習
```bash
# PPO強化学習
python main.py train-rl --algorithm ppo --episodes 500

# DQN強化学習
python main.py train-rl --algorithm dqn --episodes 500
```

## 📈 ステップ6: 結果の確認

### トレーニング結果の確認
```bash
# 出力ディレクトリの確認
ls outputs/
ls models/fine_tuned_models/
ls models/trained_models/
```

### TensorBoardでの可視化
```bash
# TensorBoardの起動
tensorboard --logdir=outputs/logs/tensorboard

# ブラウザで http://localhost:6006 を開く
```

## 🎮 応用例

### 1. カスタムドキュメントの処理
```bash
# 独自のPDF/Markdownファイルを処理
mkdir my_documents
# ファイルをmy_documentsに配置
python main.py extract my_documents/ my_outputs/ --chunk-size 1024
```

### 2. 複数モデルでのファインチューニング
```bash
# 異なるモデルでのトレーニング
python main.py train-lora --train-data data/train.jsonl --model-name gpt2
python main.py train-lora --train-data data/train.jsonl --model-name microsoft/DialoGPT-medium
```

### 3. バッチ処理での自動化
```bash
# スクリプトファイルを作成
cat > batch_training.sh << 'EOF'
#!/bin/bash
python main.py extract data/ outputs/
python main.py train-lora --train-data outputs/instruction_data.jsonl --epochs 3
python main.py train-qlora --train-data outputs/instruction_data.jsonl --epochs 2
EOF

chmod +x batch_training.sh
./batch_training.sh
```

## 📝 設定のカスタマイズ

### config.yamlの編集
```bash
# 設定ファイルを編集
nano configs/config.yaml

# 主要な設定項目：
# - device: "cuda" / "cpu" / "mps"
# - batch_size: バッチサイズ
# - learning_rate: 学習率
# - epochs: エポック数
```

### 環境変数の設定
```bash
# GPUメモリ制限
export CUDA_VISIBLE_DEVICES=0

# HuggingFaceキャッシュディレクトリ
export HF_HOME=/path/to/cache

# Weights & Biases設定
export WANDB_PROJECT=my-llm-project
```

## 🔍 次のステップ

1. **詳細ドキュメントの確認**
   - `README.md`: 全体的な概要
   - `TECHNICAL_SPECIFICATIONS.md`: 技術仕様
   - `USAGE_EXAMPLES.md`: 使用例

2. **カスタマイズの学習**
   - `configs/config.yaml`: 設定ファイル
   - `models/*/README.md`: モデル別の詳細

3. **高度な機能の活用**
   - カスタム環境の作成
   - 独自データセットの準備
   - 複数GPU環境での学習

## ❓ よくある質問

### Q1: メモリ不足エラーが発生します
```bash
# バッチサイズを減らす
python main.py train-lora --train-data data/train.jsonl --batch-size 1

# QLoRAを使用する
python main.py train-qlora --train-data data/train.jsonl
```

### Q2: GPUが認識されません
```bash
# PyTorchのGPUサポートを確認
python -c "import torch; print(torch.cuda.is_available())"

# 必要に応じてCUDA対応PyTorchを再インストール
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 学習が遅いです
```bash
# Flash Attentionをインストール
pip install flash-attn --no-build-isolation

# 混合精度学習を有効化
python main.py train-lora --train-data data/train.jsonl --fp16
```

## 🚨 トラブルシューティング

問題が発生した場合は、以下を確認してください：

1. **Python環境**: `python --version` (3.8以上)
2. **仮想環境**: `which python` (venv内か確認)
3. **GPU状態**: `nvidia-smi` (CUDA環境の場合)
4. **メモリ使用量**: `python -c "import psutil; print(psutil.virtual_memory())"`

詳細なトラブルシューティングは `README.md` の「トラブルシューティング」セクションを参照してください。