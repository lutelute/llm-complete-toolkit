# Fine-tuned Models Directory

このディレクトリには、LoRA/QLoRAによってファインチューニングされたモデルが保存されます。

## 使用方法

### LoRAファインチューニング
```bash
# LoRAファインチューニング実行
python main.py train-lora --train-data data/train.jsonl --output-dir models/fine_tuned_models/my_lora_model

# 設定ファイルを使用
python main.py train-lora --train-data data/train.jsonl --config configs/config.yaml
```

### QLoRAファインチューニング
```bash
# QLoRAファインチューニング実行
python main.py train-qlora --train-data data/train.jsonl --output-dir models/fine_tuned_models/my_qlora_model

# バッチサイズを調整（メモリ不足時）
python main.py train-qlora --train-data data/train.jsonl --batch-size 1
```

## ディレクトリ構造

各ファインチューニングされたモデルは以下の形式で保存されます：

### LoRAモデル
```
fine_tuned_models/
├── lora_model_20250115_143022/
│   ├── adapter_config.json          # LoRAアダプターの設定
│   ├── adapter_model.bin            # LoRAアダプターの重み
│   ├── training_config.yaml         # トレーニング設定
│   ├── training_log.txt             # トレーニングログ
│   ├── eval_results.json            # 評価結果
│   └── checkpoints/                 # チェックポイント
│       ├── checkpoint-100/
│       ├── checkpoint-200/
│       └── ...
```

### QLoRAモデル
```
fine_tuned_models/
├── qlora_model_20250115_143022/
│   ├── adapter_config.json          # QLoRAアダプターの設定
│   ├── adapter_model.bin            # QLoRAアダプターの重み
│   ├── quantization_config.json     # 量子化設定
│   ├── training_config.yaml         # トレーニング設定
│   ├── training_log.txt             # トレーニングログ
│   ├── eval_results.json            # 評価結果
│   └── checkpoints/                 # チェックポイント
│       ├── checkpoint-100/
│       ├── checkpoint-200/
│       └── ...
```

## モデルの使用方法

### LoRAモデルの読み込み
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ベースモデルの読み込み
base_model_name = "microsoft/DialoGPT-medium"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# LoRAアダプターの読み込み
lora_model_path = "models/fine_tuned_models/lora_model_20250115_143022"
model = PeftModel.from_pretrained(base_model, lora_model_path)
```

### QLoRAモデルの読み込み
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 量子化設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ベースモデルの読み込み（量子化あり）
base_model_name = "microsoft/DialoGPT-medium"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# QLoRAアダプターの読み込み
qlora_model_path = "models/fine_tuned_models/qlora_model_20250115_143022"
model = PeftModel.from_pretrained(base_model, qlora_model_path)
```

## 設定パラメータ

### LoRA設定
- `lora_r`: LoRAランク (デフォルト: 16)
- `lora_alpha`: LoRAアルファ (デフォルト: 32)
- `lora_dropout`: LoRAドロップアウト (デフォルト: 0.1)
- `target_modules`: 対象モジュール

### QLoRA設定
- `load_in_4bit`: 4bit量子化の有効化
- `bnb_4bit_use_double_quant`: 二重量子化の使用
- `bnb_4bit_quant_type`: 量子化タイプ ("nf4" 推奨)
- `bnb_4bit_compute_dtype`: 計算データ型

## パフォーマンス比較

| 手法 | メモリ使用量 | 学習速度 | 精度 | 用途 |
|------|-------------|----------|------|------|
| LoRA | 中 | 高 | 高 | 一般的なファインチューニング |
| QLoRA | 低 | 中 | 高 | 大規模モデル・メモリ制限時 |

## 注意事項

- LoRAモデルは元のベースモデルと組み合わせて使用する必要があります
- QLoRAモデルは量子化されたベースモデルと組み合わせて使用してください
- 学習データの品質がモデルの性能に大きく影響します
- 適切な評価データセットを用いて性能を評価してください