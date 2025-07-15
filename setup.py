#!/usr/bin/env python3
"""
LLM Complete Toolkit セットアップスクリプト
必要なパッケージの自動インストールと環境設定
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_command(command, check=True):
    """コマンドを実行"""
    logger = logging.getLogger(__name__)
    logger.info(f"実行中: {command}")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=check
    )
    
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    
    return result

def check_python_version():
    """Python バージョンチェック"""
    logger = logging.getLogger(__name__)
    
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"Python 3.8以上が必要です。現在のバージョン: {python_version}")
        sys.exit(1)
    
    logger.info(f"Python バージョン確認: {python_version.major}.{python_version.minor}.{python_version.micro} ✓")

def check_gpu_support():
    """GPU サポートの確認"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA利用可能: {torch.cuda.get_device_name(0)} ✓")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) 利用可能 ✓")
            return "mps"
        else:
            logger.info("GPU利用不可、CPUモードで実行します")
            return "cpu"
    except ImportError:
        logger.info("PyTorch未インストール、後でインストールします")
        return "unknown"

def install_requirements():
    """必要なパッケージのインストール"""
    logger = logging.getLogger(__name__)
    
    # 基本的な要件のインストール
    logger.info("基本パッケージをインストール中...")
    run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    # GPU固有のパッケージ
    device_type = check_gpu_support()
    
    if device_type == "cuda":
        logger.info("CUDA対応パッケージをインストール中...")
        try:
            # Flash Attention（オプション）
            run_command(f"{sys.executable} -m pip install flash-attn --no-build-isolation", check=False)
            logger.info("Flash Attention インストール完了")
        except:
            logger.warning("Flash Attention のインストールに失敗しました（オプション）")
    
    logger.info("パッケージインストール完了 ✓")

def setup_directories():
    """必要なディレクトリの作成"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data",
        "outputs",
        "models/base_models",
        "models/fine_tuned_models", 
        "models/trained_models",
        "logs",
        "examples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"ディレクトリ作成: {directory}")
    
    logger.info("ディレクトリ設定完了 ✓")

def create_sample_config():
    """サンプル設定ファイルの作成"""
    logger = logging.getLogger(__name__)
    
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.info("設定ファイルは既に存在します")
        return
    
    logger.info("設定ファイル確認完了 ✓")

def download_sample_model():
    """サンプルモデルのダウンロード"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("サンプルモデルをダウンロード中...")
        run_command(f"{sys.executable} scripts/download_models.py --model microsoft/DialoGPT-small")
        logger.info("サンプルモデルダウンロード完了 ✓")
    except Exception as e:
        logger.warning(f"サンプルモデルのダウンロードに失敗: {e}")

def create_sample_data():
    """サンプルデータの作成"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("サンプルデータを作成中...")
        run_command(f"{sys.executable} main.py create-samples")
        logger.info("サンプルデータ作成完了 ✓")
    except Exception as e:
        logger.warning(f"サンプルデータの作成に失敗: {e}")

def run_tests():
    """基本的なテストの実行"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("基本テストを実行中...")
        
        # インポートテスト
        test_imports = [
            "torch", "transformers", "datasets", "peft", 
            "stable_baselines3", "gymnasium", "tensorboard"
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                logger.info(f"  {module} ✓")
            except ImportError as e:
                logger.error(f"  {module} ✗ ({e})")
        
        logger.info("基本テスト完了 ✓")
        
    except Exception as e:
        logger.warning(f"テストの実行に失敗: {e}")

def main():
    """メインセットアップ処理"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 LLM Complete Toolkit セットアップ開始")
    logger.info(f"プラットフォーム: {platform.system()} {platform.release()}")
    
    try:
        # 1. Python バージョンチェック
        check_python_version()
        
        # 2. 必要なパッケージのインストール
        install_requirements()
        
        # 3. ディレクトリの作成
        setup_directories()
        
        # 4. 設定ファイルの確認
        create_sample_config()
        
        # 5. 基本テストの実行
        run_tests()
        
        # 6. サンプルデータの作成
        create_sample_data()
        
        # 7. サンプルモデルのダウンロード（オプション）
        download_sample_model()
        
        logger.info("✅ セットアップ完了！")
        logger.info("\n次のステップ:")
        logger.info("1. python main.py --help でコマンドを確認")
        logger.info("2. python scripts/download_models.py --list-popular でモデル一覧を確認")
        logger.info("3. python main.py extract data/ outputs/ でドキュメント処理をテスト")
        logger.info("4. python main.py train-lora --train-data data/train.jsonl でトレーニングを開始")
        
    except Exception as e:
        logger.error(f"セットアップエラー: {e}")
        logger.error("トラブルシューティングについては README.md を確認してください")
        sys.exit(1)

if __name__ == "__main__":
    main()