# LLM Complete Toolkit サンプル

これはLLM Complete Toolkitのサンプル文書です。

## 機能概要

このツールキットは以下の機能を提供します：

### ドキュメント処理
- PDFファイルの解析とテキスト抽出
- Markdownファイルの解析
- LM Studio用データ形式への変換

### 機械学習手法
- **転移学習**: LoRA/QLoRAによる効率的ファインチューニング
- **強化学習**: PPO/DQNによるエージェント学習

## 使用方法

```bash
# ドキュメント抽出
python main.py extract data/input data/output

# LoRAファインチューニング
python main.py train-lora --train-data data/train.jsonl

# 強化学習
python main.py train-rl --algorithm ppo
```

## まとめ

このツールキットを使用することで、文書処理から機械学習まで一貫したワークフローが実現できます。
