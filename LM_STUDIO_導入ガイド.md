# LM Studio モデル導入ガイド

## 概要
学習済みのGFMモデルをLM Studioで使用するための完全ガイドです。

## 手順1: モデルの場所を確認

学習済みモデルは以下の場所に保存されています：
```
/Users/shigenoburyuto/Documents/GitHub/llm-complete-toolkit/models/lm_studio_models/GFM-DialoGPT-small-final/
```

## 手順2: LM Studioでモデルを読み込む

### 方法A: フォルダから直接読み込み（推奨）
1. LM Studioを起動
2. 「Load Model」をクリック
3. 「Load from folder」を選択
4. 以下のフォルダを選択：
   ```
   /Users/shigenoburyuto/Documents/GitHub/llm-complete-toolkit/models/lm_studio_models/GFM-DialoGPT-small-final
   ```

### 方法B: モデルをコピーしてから読み込み
もしフォルダから直接読み込みができない場合：

1. **モデルフォルダをコピー**
   ```bash
   # LM Studioのデフォルトモデルフォルダにコピー
   cp -r "/Users/shigenoburyuto/Documents/GitHub/llm-complete-toolkit/models/lm_studio_models/GFM-DialoGPT-small-final" ~/Documents/LMStudio/models/
   ```

2. **LM Studioで読み込み**
   - LM Studioを起動
   - モデル一覧から「GFM-DialoGPT-small-final」を選択

## 手順3: モデル設定

### 推奨パラメータ
- **Temperature**: 0.7
- **Top P**: 0.9  
- **Max Tokens**: 512
- **Repeat Penalty**: 1.1

### GFMテスト用設定
- **System Prompt**: "あなたはGFM（Grid-Forming Inverter）の専門家です。技術的な質問に正確に答えてください。"
- **Context Length**: 1024

## 手順4: モデルテスト

### 基本動作確認
まず簡単な挨拶でテスト：
```
こんにちは。調子はどうですか？
```

### GFM知識テスト
以下の質問でGFM知識を確認：
```
GFMとは何ですか？
```

```
グリッドフォーミングインバーターの主な機能を教えてください。
```

## トラブルシューティング

### エラー: "No LM Runtime found for model format"
これは解決済みです。最新の変換済みモデル（`GFM-DialoGPT-small-final`）を使用してください。

### モデルが認識されない場合
1. フォルダ内に以下のファイルがあることを確認：
   - `pytorch_model.bin` (474.7 MB)
   - `config.json`
   - `tokenizer.json`
   - `generation_config.json`

2. LM Studioを再起動

### 読み込みが遅い場合
- Apple Silicon (M4) では最初の読み込みに時間がかかることがあります
- 「device: mps」で最適化されているので待機してください

## ファイル構成
```
GFM-DialoGPT-small-final/
├── pytorch_model.bin          # メインモデル (474.7 MB)
├── config.json               # モデル設定
├── tokenizer.json            # トークナイザー (3.4 MB)
├── generation_config.json    # 生成パラメータ
├── model_info.json          # LM Studio用情報
├── README.md                # 使用方法
└── その他設定ファイル
```

## 次のステップ
モデルが正常に読み込めたら、GFM評価ガイド（`GFM_EVALUATION_GUIDE.md`）を参考にして詳細なテストを実行してください。