# 📚 LLM Complete Toolkit 実行マニュアル

## 目次
1. [環境準備](#環境準備)
2. [データ準備](#データ準備)
3. [学習実行](#学習実行)
4. [学習後のモデル統合](#学習後のモデル統合)
5. [LM Studio確認プロセス](#lm-studio確認プロセス)
6. [GFM知識評価テスト](#gfm知識評価テスト)
7. [トラブルシューティング](#トラブルシューティング)

---

## 環境準備

### 1. 仮想環境の作成と有効化
```bash
# 仮想環境作成
python3 -m venv venv

# 仮想環境有効化
source venv/bin/activate

# 依存関係インストール
pip install torch transformers peft datasets psutil jsonlines pdfplumber pypdf2 markdown
```

### 2. ディレクトリ構造確認
```
llm-complete-toolkit/
├── main.py
├── training_data/
│   ├── raw/GFM/          # 学習用生データ
│   └── datasets/         # 処理済みデータ
├── models/
│   ├── base_models/      # ベースモデル
│   ├── fine_tuned_models/ # LoRAアダプター
│   └── lm_studio_models/ # マージ済みモデル
└── configs/config.yaml
```

---

## データ準備

### 1. 生データ配置
```bash
# GFM関連データをraw/GFM/に配置
mkdir -p training_data/raw/GFM
# PDFファイル、Markdownファイル、JSONファイルを配置
```

### 2. 学習データ生成
```bash
# 仮想環境有効化
source venv/bin/activate

# 学習用データ抽出・変換
python main.py extract-training-data \
  --input-dir training_data/raw/GFM \
  --output-dir training_data/processed \
  --format all \
  --output-format jsonl \
  --instruction-format \
  --split-data \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

**期待される出力:**
```
訓練データ: training_data/datasets/training/train.jsonl
検証データ: training_data/datasets/validation/eval.jsonl
テストデータ: training_data/datasets/test/test.jsonl
```

---

## 学習実行

### 1. 高速化LoRA学習
```bash
# 仮想環境有効化
source venv/bin/activate

# 最適化されたLoRA学習実行
python main.py train-lora \
  --train-data training_data/datasets/training/train.jsonl \
  --eval-data training_data/datasets/validation/eval.jsonl \
  --model-name microsoft/DialoGPT-small \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --output-dir models/fine_tuned_models/GFM-DialoGPT-small \
  --verbose
```

### 2. 学習パラメータ説明
- `--epochs 3`: 3エポック学習（時間に応じて調整）
- `--batch-size 4`: バッチサイズ（メモリに応じて調整）
- `--learning-rate 5e-5`: 学習率（推奨値）
- `--verbose`: 詳細ログ出力

### 3. 学習状況モニタリング
学習中に以下が表示されます：
- メモリ使用状況
- 学習進捗
- パフォーマンス統計
- 自動最適化情報

---

## 学習後のモデル統合

### 1. LoRAアダプターをベースモデルにマージ
```bash
# LoRAアダプターをベースモデルとマージ
python main.py merge-lora \
  --base-model models/base_models/microsoft_DialoGPT-small \
  --lora-adapter models/fine_tuned_models/GFM-DialoGPT-small \
  --output models/lm_studio_models/GFM-DialoGPT-small-merged \
  --device auto
```

### 2. マージ結果確認
```bash
# マージ済みモデルの確認
ls -la models/lm_studio_models/GFM-DialoGPT-small-merged/
```

**期待されるファイル:**
- `pytorch_model.bin` または `model.safetensors`
- `tokenizer.json`
- `config.json`
- `generation_config.json`

---

## LM Studio確認プロセス

### 1. LM Studioのインストール
1. [LM Studio公式サイト](https://lmstudio.ai/)からダウンロード
2. インストール後、LM Studioを起動

### 2. モデルの読み込み
1. **「Load Model」** を選択
2. **「Load from folder」** を選択
3. マージ済みモデルディレクトリを選択:
   ```
   models/lm_studio_models/GFM-DialoGPT-small-merged/
   ```

### 3. 基本動作確認
#### 学習前モデル（ベースライン）
```
プロンプト: "GFMとは何ですか？"
期待する回答: 一般的な回答または不正確な回答
```

#### 学習後モデル（GFM専門化）
```
プロンプト: "GFMとは何ですか？"
期待する回答: Grid-Forming Inverterに関する専門的な回答
```

### 4. 推論設定
**推奨設定:**
- Temperature: 0.7
- Top P: 0.9
- Max Tokens: 512
- Repeat Penalty: 1.1

---

## GFM知識評価テスト

### 1. 基本評価質問リスト

#### 🔰 初級（略語レベルの基本理解）
```
1. GFMの主な機能とその系統内での役割は？
2. GFLとGFMの構造的・動作的な違いは？
3. VSGとしてのGFMの挙動は何を模倣しているか？
```

#### ⚙️ 中級（略語＋制御理論）
```
4. GFMにおけるDroop制御のf-P, V-Q関係を説明せよ。
5. GFMがISLやBSにおいて有効とされる理由は？
6. VI（仮想慣性）はGFMのどの挙動に対応しているか？またその効果は？
```

#### 🧠 上級（略語＋系統応用）
```
7. GFMが高PEN状態の電力系統に与える影響は？特にFRとVSRに注目して論ぜよ。
8. SGとGFMが混在するHYB構成で必要なCCTや協調機構は？
9. GFM制御の代表的手法（例：Droop, VSM, DVCなど）とその特性は？
10. GFMにおけるDroop係数やVIゲイン設計に必要なTSCやSSMに関する条件とは？
```

### 2. 評価方法

#### A. 学習前（ベースライン）測定
```bash
# ベースモデルでの評価
python main.py train-lora \
  --model-name microsoft/DialoGPT-small \
  --output-dir models/baseline_test \
  --epochs 0  # 学習なし、評価のみ
```

#### B. 学習後（専門化）測定
```bash
# 学習済みモデルでの評価
# LM Studioで学習済みモデルを使用
```

#### C. 評価基準
- **正確性**: 専門用語の正しい理解
- **完全性**: 回答の網羅性
- **専門性**: 技術的詳細の深さ
- **実用性**: 実際の応用に関する知識

### 3. 評価結果記録テンプレート

```markdown
## GFM知識評価結果

### 基本情報
- 評価日時: [日時]
- モデル: [モデル名]
- 学習データ: [データ概要]

### 評価結果
| 質問No | 質問レベル | 学習前スコア | 学習後スコア | 改善度 |
|--------|------------|-------------|-------------|--------|
| 1      | 初級       | [0-5]       | [0-5]       | [差分] |
| 2      | 初級       | [0-5]       | [0-5]       | [差分] |
| ...    | ...        | ...         | ...         | ...    |

### 総合評価
- 初級平均: [前] → [後] (改善: [差分])
- 中級平均: [前] → [後] (改善: [差分])
- 上級平均: [前] → [後] (改善: [差分])
- 全体平均: [前] → [後] (改善: [差分])

### 特記事項
- 特に改善された領域: [記載]
- 改善が必要な領域: [記載]
- 追加学習の提案: [記載]
```

---

## トラブルシューティング

### 1. メモリ不足エラー
```bash
# バッチサイズを減らす
--batch-size 2

# または勾配蓄積を増やす
--gradient-accumulation-steps 8
```

### 2. CUDA/MPS関連エラー
```bash
# CPUモードで実行
--device cpu

# またはMPS無効化
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 3. トークナイザーエラー
```bash
# キャッシュクリア
rm -rf cache/
```

### 4. 学習が開始しない
```bash
# 依存関係を再インストール
pip install --upgrade torch transformers peft datasets
```

### 5. LM Studioでモデルが読み込めない
1. モデルファイルの存在確認
2. ファイル権限の確認
3. LM Studioの再起動

---

## 高速化機能

### 1. キャッシュ機能
- **初回実行**: 通常の処理時間
- **2回目以降**: 大幅な高速化（キャッシュ利用）

### 2. 動的最適化
- **自動バッチサイズ調整**: メモリ使用量に応じて自動調整
- **メモリクリーンアップ**: 定期的な自動メモリ解放
- **パフォーマンス監視**: リアルタイムでの最適化

### 3. 安定性機能
- **エラー自動回復**: OOM時の自動調整
- **フォールバック機能**: 高速化失敗時の安全な処理
- **詳細ログ**: 問題特定のための詳細情報

---

## 実行例

### 完全な実行フロー
```bash
# 1. 環境準備
source venv/bin/activate

# 2. データ準備
python main.py extract-training-data \
  --input-dir training_data/raw/GFM \
  --split-data --instruction-format

# 3. 学習実行
python main.py train-lora \
  --train-data training_data/datasets/training/train.jsonl \
  --eval-data training_data/datasets/validation/eval.jsonl \
  --model-name microsoft/DialoGPT-small \
  --epochs 3 --batch-size 4 --learning-rate 5e-5 \
  --output-dir models/fine_tuned_models/GFM-DialoGPT-small \
  --verbose

# 4. モデルマージ
python main.py merge-lora \
  --base-model models/base_models/microsoft_DialoGPT-small \
  --lora-adapter models/fine_tuned_models/GFM-DialoGPT-small \
  --output models/lm_studio_models/GFM-DialoGPT-small-merged

# 5. LM Studioで確認
# - LM Studioを起動
# - マージ済みモデルを読み込み
# - GFM評価質問で性能確認
```

このマニュアルに従って実行することで、GFM専門知識を持つカスタマイズされたLLMを作成し、LM Studioで評価することができます。