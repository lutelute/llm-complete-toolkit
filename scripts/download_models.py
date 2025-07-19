#!/usr/bin/env python3
"""
Hugging Face モデルダウンロードスクリプト
事前学習済みモデルをダウンロードし、base_modelsフォルダに保存する
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_model(model_name: str, output_dir: str = "models/base_models", 
                  cache_dir: Optional[str] = None) -> bool:
    """
    Hugging Faceからモデルをダウンロード
    
    Args:
        model_name: モデル名 (例: "microsoft/DialoGPT-medium")
        output_dir: 出力ディレクトリ
        cache_dir: キャッシュディレクトリ
        
    Returns:
        bool: ダウンロード成功の場合True
    """
    logger = logging.getLogger(__name__)
    logger.info(f"モデルダウンロード開始: {model_name}")
    
    try:
        # 出力ディレクトリの作成
        output_path = Path(output_dir) / model_name.replace("/", "_")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # モデルファイルのダウンロード
        logger.info(f"モデルファイルをダウンロード中: {model_name}")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path),
            cache_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        # トークナイザーとコンフィグの確認
        logger.info("モデルの整合性を確認中...")
        tokenizer = AutoTokenizer.from_pretrained(str(output_path))
        config = AutoConfig.from_pretrained(str(output_path))
        
        # モデル情報の保存
        info_file = output_path / "model_info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Model Type: {config.model_type}\n")
            f.write(f"Architecture: {config.architectures}\n")
            f.write(f"Vocabulary Size: {config.vocab_size}\n")
            f.write(f"Max Position: {getattr(config, 'max_position_embeddings', 'N/A')}\n")
            f.write(f"Hidden Size: {getattr(config, 'hidden_size', 'N/A')}\n")
            f.write(f"Number of Layers: {getattr(config, 'num_hidden_layers', 'N/A')}\n")
            f.write(f"Number of Attention Heads: {getattr(config, 'num_attention_heads', 'N/A')}\n")
            f.write(f"Tokenizer Type: {type(tokenizer).__name__}\n")
            f.write(f"Tokenizer Vocab Size: {len(tokenizer)}\n")
        
        logger.info(f"モデルダウンロード完了: {output_path}")
        logger.info(f"モデル情報: {info_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"モデルダウンロードエラー: {e}")
        return False

def download_multiple_models(model_list: List[str], output_dir: str = "models/base_models",
                           cache_dir: Optional[str] = None) -> dict:
    """
    複数のモデルをダウンロード
    
    Args:
        model_list: モデル名のリスト
        output_dir: 出力ディレクトリ
        cache_dir: キャッシュディレクトリ
        
    Returns:
        dict: ダウンロード結果 {model_name: success_status}
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    for model_name in model_list:
        logger.info(f"モデル {model_name} のダウンロードを開始...")
        success = download_model(model_name, output_dir, cache_dir)
        results[model_name] = success
        
        if success:
            logger.info(f"✓ {model_name} のダウンロードが完了しました")
        else:
            logger.error(f"✗ {model_name} のダウンロードに失敗しました")
    
    return results

def get_popular_models() -> List[str]:
    """よく使われるモデルのリストを取得"""
    return [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-large",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "distilgpt2",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-j-6B",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-3b",
        "microsoft/CodeGPT-small-py",
        "Salesforce/codegen-350M-mono",
        "huggingface/CodeBERTa-small-v1"
    ]

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description='Hugging Face モデルダウンロードツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一モデルのダウンロード
  python download_models.py --model microsoft/DialoGPT-medium
  
  # 複数モデルのダウンロード
  python download_models.py --model gpt2 --model microsoft/DialoGPT-small
  
  # 人気モデルのリスト表示
  python download_models.py --list-popular
  
  # 人気モデルをすべてダウンロード
  python download_models.py --download-popular
        """
    )
    
    parser.add_argument('--model', type=str, action='append', dest='models',
                       help='ダウンロードするモデル名（複数指定可能）')
    parser.add_argument('--output-dir', type=str, default='models/base_models',
                       help='出力ディレクトリ (default: models/base_models)')
    parser.add_argument('--cache-dir', type=str,
                       help='キャッシュディレクトリ')
    parser.add_argument('--list-popular', action='store_true',
                       help='人気モデルのリストを表示')
    parser.add_argument('--download-popular', action='store_true',
                       help='人気モデルをすべてダウンロード')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細ログを表示')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 人気モデルのリスト表示
    if args.list_popular:
        popular_models = get_popular_models()
        print("人気モデル一覧:")
        for i, model in enumerate(popular_models, 1):
            print(f"{i:2d}. {model}")
        return
    
    # ダウンロード対象の決定
    models_to_download = []
    
    if args.download_popular:
        models_to_download.extend(get_popular_models())
    
    if args.models:
        models_to_download.extend(args.models)
    
    if not models_to_download:
        logger.error("ダウンロードするモデルが指定されていません")
        logger.info("--model で個別指定するか、--download-popular で人気モデルをダウンロード")
        parser.print_help()
        sys.exit(1)
    
    # 重複を除去
    models_to_download = list(set(models_to_download))
    
    logger.info(f"ダウンロード対象モデル数: {len(models_to_download)}")
    for model in models_to_download:
        logger.info(f"  - {model}")
    
    # ダウンロード実行
    results = download_multiple_models(
        models_to_download, 
        args.output_dir, 
        args.cache_dir
    )
    
    # 結果の表示
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"\n=== ダウンロード結果 ===")
    logger.info(f"成功: {success_count}/{total_count}")
    
    if success_count < total_count:
        logger.info("失敗したモデル:")
        for model, success in results.items():
            if not success:
                logger.info(f"  - {model}")
    
    logger.info("処理完了")

if __name__ == "__main__":
    main()