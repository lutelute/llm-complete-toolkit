# 学習用データ格納フォルダ

このディレクトリは、LLMトレーニング用のデータを格納するためのフォルダです。

## 📁 フォルダ構造

```
training_data/
├── raw/                    # 生データ格納（全形式混在OK）
│   ├── document1.pdf      # PDFファイル
│   ├── notes.md           # Markdownファイル
│   ├── data.json          # JSONファイル
│   ├── text_file.txt      # テキストファイル
│   └── mixed_files...     # 複数形式のファイルが混在
├── processed/             # 処理済みデータ
│   ├── training_data.jsonl        # 変換されたデータ
│   └── instruction_data.jsonl     # インストラクション形式
└── datasets/              # 学習用データセット（分割済み）
    ├── training/          # 訓練用データ
    │   └── train.jsonl
    ├── validation/        # 検証用データ
    │   └── eval.jsonl
    └── test/              # テスト用データ
        └── test.jsonl
```

## 📄 対応ファイル形式

### 入力ファイル（`raw/`フォルダに配置）
- **PDF**: `.pdf`
- **Markdown**: `.md`, `.markdown`
- **テキスト**: `.txt`
- **JSON**: `.json`, `.jsonl`

### 出力ファイル（自動生成）
- **JSONL**: 学習用データ形式
- **CSV**: 分析用データ形式
- **TXT**: プレーンテキスト形式

## 🚀 使用方法

### 1. データの配置
```bash
# 全てのファイルを一つのフォルダに配置（形式混在OK）
cp /path/to/your/mixed/files/* training_data/raw/

# または個別にコピー
cp /path/to/your/documents/*.pdf training_data/raw/
cp /path/to/your/notes/*.md training_data/raw/
cp /path/to/your/data/*.json training_data/raw/
cp /path/to/your/texts/*.txt training_data/raw/

# フォルダごとコピーも可能
cp -r /path/to/your/document_folder/* training_data/raw/
```

### 2. データの処理
```bash
# 全形式のデータを自動処理（推奨）
python main.py extract-training-data

# データ分割付き処理
python main.py extract-training-data --split-data --instruction-format

# 特定の形式のみ処理
python main.py extract-training-data --format pdf
python main.py extract-training-data --format markdown
python main.py extract-training-data --format text
python main.py extract-training-data --format json

# 出力形式を指定
python main.py extract-training-data --output-format jsonl
python main.py extract-training-data --output-format csv
```

### 3. 学習の実行
```bash
# 処理済みデータで学習
python main.py train-lora --train-data training_data/datasets/training/train.jsonl

# 検証データ付きで学習
python main.py train-lora \
  --train-data training_data/datasets/training/train.jsonl \
  --eval-data training_data/datasets/validation/eval.jsonl
```

## 📋 データ形式の例

### PDFファイル
```
training_data/raw/pdf/
├── document1.pdf
├── document2.pdf
└── research_paper.pdf
```

### Markdownファイル
```
training_data/raw/markdown/
├── tutorial.md
├── documentation.md
└── notes.markdown
```

### JSONファイル
```json
{
  "instruction": "質問の内容",
  "input": "追加の入力（オプション）",
  "output": "期待される回答"
}
```

### JSONLファイル
```jsonl
{"instruction": "質問1", "input": "", "output": "回答1"}
{"instruction": "質問2", "input": "", "output": "回答2"}
```

## 🔄 処理フロー

1. **データ配置**: `raw/`フォルダに元データを配置
2. **抽出処理**: 各形式からテキストを抽出 → `processed/extracted/`
3. **チャンク分割**: 適切なサイズに分割 → `processed/chunks/`
4. **フォーマット変換**: 学習用形式に変換 → `processed/formatted/`
5. **データセット作成**: 訓練/検証/テスト用に分割 → `datasets/`

## ⚙️ 設定オプション

### `configs/config.yaml`での設定
```yaml
training_data:
  raw_data_dir: "./training_data/raw"
  processed_data_dir: "./training_data/processed"
  datasets_dir: "./training_data/datasets"
  
  # チャンク設定
  chunk_size: 512
  chunk_overlap: 50
  
  # データ分割比率
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
  # フォーマット設定
  output_format: "jsonl"
  instruction_format: true
```

## 🔍 データ確認

### 処理済みデータの確認
```bash
# 処理済みファイルの一覧
ls -la training_data/processed/

# 生成されたデータセットの確認
head -5 training_data/datasets/training/train.jsonl

# データ統計の確認
python -c "
import json
with open('training_data/datasets/training/train.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    print(f'Total samples: {len(data)}')
    print(f'Average text length: {sum(len(d[\"output\"]) for d in data) / len(data):.1f}')
"
```

## 🚨 注意事項

1. **大きなファイル**: 大量のデータの場合、処理に時間がかかる可能性があります
2. **メモリ使用量**: 大きなファイルの処理時はメモリ使用量に注意してください
3. **エンコーディング**: 日本語ファイルはUTF-8エンコーディングを推奨します
4. **バックアップ**: 重要なデータは事前にバックアップを取ってください

## 💡 ヒント

- 小さなデータセットでまずテストしてから大量データを処理する
- 処理時間を短縮するため、不要なファイルは事前に除外する
- 定期的に中間結果を確認して、期待通りの処理が行われているか確認する