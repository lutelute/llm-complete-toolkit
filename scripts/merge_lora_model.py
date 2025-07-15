#!/usr/bin/env python3
"""
LoRAアダプターをベースモデルにマージしてLM Studio用のモデルを作成
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def merge_lora_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device: str = "auto"
) -> None:
    """
    LoRAアダプターをベースモデルにマージ
    
    Args:
        base_model_path: ベースモデルのパス
        lora_adapter_path: LoRAアダプターのパス
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
    logger.info(f"ベースモデル読み込み: {base_model_path}")
    
    # ベースモデルの読み込み
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True
    )
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    logger.info(f"LoRAアダプター読み込み: {lora_adapter_path}")
    
    # LoRAアダプターの読み込み
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    logger.info("LoRAアダプターをベースモデルにマージ中...")
    
    # アダプターをベースモデルにマージ
    model = model.merge_and_unload()
    
    # 出力ディレクトリの作成
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"マージ済みモデル保存中: {output_path}")
    
    # マージ済みモデルの保存
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    # トークナイザーの保存
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"マージ完了: {output_path}")
    logger.info("このモデルはLM Studioで使用可能です")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="LoRAアダプターをベースモデルにマージしてLM Studio用のモデルを作成"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="ベースモデルのパス"
    )
    
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="LoRAアダプターのパス"
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
        merge_lora_model(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter,
            output_path=args.output,
            device=args.device
        )
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()