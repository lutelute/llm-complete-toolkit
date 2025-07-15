# Base Models Directory

このディレクトリには、Hugging Faceからダウンロードした未学習（事前学習済み）モデルが保存されます。

## 使用方法

### モデルのダウンロード
```bash
# 単一モデルのダウンロード
python scripts/download_models.py --model microsoft/DialoGPT-medium

# 複数モデルのダウンロード
python scripts/download_models.py --model gpt2 --model microsoft/DialoGPT-small

# 人気モデルの一覧表示
python scripts/download_models.py --list-popular

# 人気モデルをすべてダウンロード
python scripts/download_models.py --download-popular
```

## ディレクトリ構造

各モデルは以下の形式で保存されます：
```
base_models/
├── microsoft_DialoGPT-medium/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── model_info.txt
├── gpt2/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── model_info.txt
└── ...
```

## 推奨モデル

### 対話生成
- `microsoft/DialoGPT-small` (117M パラメータ)
- `microsoft/DialoGPT-medium` (345M パラメータ)
- `microsoft/DialoGPT-large` (762M パラメータ)

### 一般的なテキスト生成
- `gpt2` (124M パラメータ)
- `gpt2-medium` (355M パラメータ)
- `gpt2-large` (774M パラメータ)
- `distilgpt2` (82M パラメータ、軽量版)

### 効率的なモデル
- `facebook/opt-125m` (125M パラメータ)
- `facebook/opt-350m` (350M パラメータ)
- `EleutherAI/gpt-neo-125M` (125M パラメータ)

### コード生成
- `microsoft/CodeGPT-small-py` (124M パラメータ)
- `Salesforce/codegen-350M-mono` (350M パラメータ)

## 注意事項

- モデルファイルは大きいため、十分な容量を確保してください
- 初回ダウンロード時は時間がかかる場合があります
- モデルの利用にはそれぞれのライセンスに従ってください