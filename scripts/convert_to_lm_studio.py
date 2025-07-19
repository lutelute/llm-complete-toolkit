#!/usr/bin/env python3
"""
LM Studio専用の互換性変換スクリプト
"""

import argparse
import logging
import sys
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def convert_to_lm_studio(
    model_path: str,
    output_path: str,
    device: str = "auto"
) -> None:
    """
    モデルをLM Studio互換形式に変換
    
    Args:
        model_path: 変換元モデルのパス
        output_path: 出力パス
        device: デバイス
    """
    logger = logging.getLogger(__name__)
    
    # デバイス設定
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"使用デバイス: {device}")
    logger.info(f"モデル読み込み: {model_path}")
    
    # モデルとトークナイザーの読み込み
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return
    
    # 出力ディレクトリの作成
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"LM Studio互換形式で変換中: {output_path}")
    
    # PyTorch形式で保存（safe_tensors無効）
    model.save_pretrained(
        output_path,
        safe_serialization=False,  # PyTorch .bin形式で保存
        max_shard_size="2GB"
    )
    
    # トークナイザーの保存
    tokenizer.save_pretrained(output_path)
    
    # 設定ファイルのコピー・調整
    config.save_pretrained(output_path)
    
    # LM Studio用のgeneration_config.jsonを作成
    generation_config = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None
    }
    
    generation_config_path = output_dir / "generation_config.json"
    with open(generation_config_path, 'w', encoding='utf-8') as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)
    
    # LM Studio用のmodel_info.jsonを作成
    model_info = {
        "model_type": config.model_type,
        "architectures": config.architectures,
        "vocab_size": config.vocab_size,
        "hidden_size": getattr(config, 'hidden_size', getattr(config, 'n_embd', None)),
        "num_attention_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', None)),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None)),
        "converted_for": "LM Studio",
        "conversion_date": str(torch.utils.data.get_worker_info() or "unknown")
    }
    
    model_info_path = output_dir / "model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    # 必要に応じてREADMEを作成
    readme_content = f"""# LM Studio互換モデル

このモデルはLM Studio用に変換されました。

## モデル情報
- 元モデル: {model_path}
- モデルタイプ: {config.model_type}
- 語彙サイズ: {config.vocab_size}

## 使用方法
1. LM Studioを起動
2. "Load Model" → "Load from folder"を選択
3. このフォルダを選択: {output_path}

## 推奨設定
- Temperature: 0.7
- Top P: 0.9
- Max Tokens: 512
- Repeat Penalty: 1.1
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # ファイル一覧を確認
    logger.info("変換完了! 作成されたファイル:")
    for file_path in sorted(output_dir.iterdir()):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_path.name}: {size_mb:.1f} MB")
    
    logger.info(f"LM Studio互換変換完了: {output_path}")
    logger.info("LM Studioでフォルダから読み込んでください")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="モデルをLM Studio互換形式に変換"
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
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="使用デバイス"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        convert_to_lm_studio(
            model_path=args.model_path,
            output_path=args.output,
            device=args.device
        )
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()