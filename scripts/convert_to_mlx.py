#!/usr/bin/env python3
"""
MLX形式への変換スクリプト（Apple Silicon最適化）
"""

import argparse
import logging
import sys
import json
import shutil
from pathlib import Path
from typing import Optional

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def convert_to_mlx(
    model_path: str,
    output_path: str
) -> None:
    """
    モデルをMLX形式に変換
    
    Args:
        model_path: 変換元モデルのパス
        output_path: 出力パス
    """
    logger = logging.getLogger(__name__)
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import convert
        logger.info("MLXライブラリが利用可能です")
    except ImportError:
        logger.error("MLXライブラリがインストールされていません")
        logger.info("インストール方法:")
        logger.info("pip install mlx-lm")
        return
    
    logger.info(f"モデル読み込み: {model_path}")
    logger.info(f"MLX形式で変換中: {output_path}")
    
    # 出力ディレクトリの作成
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # MLX形式に変換
        convert(
            hf_path=model_path,
            mlx_path=output_path,
            quantize=False,  # 量子化なし
            q_group_size=64,
            q_bits=4
        )
        
        logger.info("MLX変換完了!")
        
        # MLX用のREADMEを作成
        readme_content = f"""# MLX形式モデル

このモデルはMLX（Apple Silicon最適化）用に変換されました。

## モデル情報
- 元モデル: {model_path}
- MLX形式: Apple Silicon (M1/M2/M3/M4) 最適化済み
- 推論速度: 大幅に高速化

## 使用方法

### Python API
```python
from mlx_lm import load, generate

# モデル読み込み
model, tokenizer = load("{output_path}")

# テキスト生成
prompt = "GFMとは何ですか？"
response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
print(response)
```

### コマンドライン
```bash
# テキスト生成
python -m mlx_lm.generate --model {output_path} --prompt "GFMとは何ですか？"

# チャット形式
python -m mlx_lm.chat --model {output_path}
```

## 推奨設定
- Max Tokens: 512
- Temperature: 0.7
- Top P: 0.9

## パフォーマンス
- Apple Silicon上で最高速度
- メモリ効率が大幅改善
- CPUとGPUの統合メモリを効率活用
"""
        
        readme_path = output_dir / "README_MLX.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # ファイル一覧を確認
        logger.info("作成されたファイル:")
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.name}: {size_mb:.1f} MB")
        
        logger.info(f"MLX変換完了: {output_path}")
        logger.info("MLXライブラリで高速推論が可能です")
        
    except Exception as e:
        logger.error(f"MLX変換エラー: {e}")
        logger.info("代替方法: mlx-lmパッケージを使用してください")
        logger.info(f"コマンド: python -m mlx_lm.convert --hf-path {model_path} --mlx-path {output_path}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="モデルをMLX形式に変換（Apple Silicon最適化）"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="変換元モデルのパス"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力パス"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        convert_to_mlx(
            model_path=args.model_path,
            output_path=args.output
        )
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()