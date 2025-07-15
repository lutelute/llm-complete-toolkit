#!/usr/bin/env python3
"""
LLM Complete Toolkit - 統合メインランチャー
PDFやMarkdownからデータを抽出し、強化学習・転移学習でLLMをトレーニングする統合ツールキット
"""

import argparse
import sys
import subprocess
import json
import jsonlines
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from document_processing.parsers.pdf_parser import PDFParser
from document_processing.parsers.markdown_parser import MarkdownParser
from document_processing.converters.lm_studio_converter import LMStudioConverter
from shared_utils.file_utils import get_files_by_extension, validate_input_directory, create_output_directory
from shared_utils.training_utils import setup_logging


def setup_project_logging():
    """プロジェクト用ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_command(command: list, cwd: Path = None):
    """コマンドを実行"""
    logger = logging.getLogger(__name__)
    logger.info(f"実行中: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"出力: {result.stdout}")
        
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"コマンド実行エラー: {e}")
        if e.stdout:
            logger.error(f"標準出力: {e.stdout}")
        if e.stderr:
            logger.error(f"標準エラー: {e.stderr}")
        raise


def extract_documents(args):
    """PDFやMarkdownからドキュメント抽出"""
    logger = logging.getLogger(__name__)
    logger.info("ドキュメント抽出を開始します")
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 入力ディレクトリの検証
    if not validate_input_directory(input_dir):
        logger.error("入力ディレクトリの検証に失敗しました")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    create_output_directory(output_dir)
    
    all_documents = []
    
    # PDFファイルの処理
    pdf_files = get_files_by_extension(input_dir, '.pdf')
    if pdf_files:
        logger.info(f"PDFファイル {len(pdf_files)}件を処理中...")
        pdf_parser = PDFParser()
        
        for pdf_file in pdf_files:
            logger.info(f"処理中: {pdf_file}")
            try:
                documents = pdf_parser.parse(pdf_file)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"PDFファイル処理エラー {pdf_file}: {e}")
    
    # Markdownファイルの処理
    md_files = get_files_by_extension(input_dir, '.md')
    if md_files:
        logger.info(f"Markdownファイル {len(md_files)}件を処理中...")
        md_parser = MarkdownParser()
        
        for md_file in md_files:
            logger.info(f"処理中: {md_file}")
            try:
                documents = md_parser.parse(md_file)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Markdownファイル処理エラー {md_file}: {e}")
    
    if not all_documents:
        logger.error("処理可能なドキュメントが見つかりませんでした")
        sys.exit(1)
    
    # LM Studio用データに変換
    logger.info("LM Studio用データに変換中...")
    converter = LMStudioConverter(
        chunk_size=args.chunk_size,
        output_format=args.format
    )
    
    converted_data = converter.convert(all_documents)
    
    # インストラクション形式の変換（オプション）
    if args.instruction_format:
        logger.info("インストラクション形式にも変換中...")
        instruction_data = converter.convert_for_instruction_tuning(all_documents)
        
        instruction_output = output_dir / f"instruction_data.{args.format}"
        converter.save_instruction_format(instruction_data, instruction_output)
    
    # ファイル出力
    output_file = output_dir / f"training_data.{args.format}"
    converter.save(converted_data, output_file)
    
    logger.info(f"ドキュメント抽出完了:")
    logger.info(f"  処理ドキュメント数: {len(all_documents)}")
    logger.info(f"  生成チャンク数: {len(converted_data)}")
    logger.info(f"  出力ファイル: {output_file}")


def launch_lora_training(args):
    """LoRAトレーニングを開始"""
    logger = logging.getLogger(__name__)
    logger.info("LoRAファインチューニングを開始します")
    
    if not args.train_data:
        logger.error("--train-dataが必要です")
        sys.exit(1)
    
    command = [
        sys.executable, "scripts/train_lora.py",
        "--train-data", args.train_data
    ]
    
    if args.config:
        command.extend(["--config", args.config])
    if args.eval_data:
        command.extend(["--eval-data", args.eval_data])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.model_name:
        command.extend(["--model-name", args.model_name])
    if args.epochs:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        command.extend(["--learning-rate", str(args.learning_rate)])
    if args.verbose:
        command.append("-v")
    
    run_command(command)


def launch_qlora_training(args):
    """QLoRAトレーニングを開始"""
    logger = logging.getLogger(__name__)
    logger.info("QLoRAファインチューニングを開始します")
    
    if not args.train_data:
        logger.error("--train-dataが必要です")
        sys.exit(1)
    
    command = [
        sys.executable, "scripts/train_qlora.py",
        "--train-data", args.train_data
    ]
    
    if args.config:
        command.extend(["--config", args.config])
    if args.eval_data:
        command.extend(["--eval-data", args.eval_data])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.model_name:
        command.extend(["--model-name", args.model_name])
    if args.epochs:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        command.extend(["--learning-rate", str(args.learning_rate)])
    if args.verbose:
        command.append("-v")
    
    run_command(command)


def launch_rl_training(args):
    """強化学習トレーニングを開始"""
    logger = logging.getLogger(__name__)
    logger.info(f"強化学習トレーニングを開始します: {args.algorithm}")
    
    command = [
        sys.executable, "scripts/train_rl.py",
        "--algorithm", args.algorithm
    ]
    
    if args.config:
        command.extend(["--config", args.config])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.episodes:
        command.extend(["--episodes", str(args.episodes)])
    if args.verbose:
        command.append("-v")
    
    run_command(command)


def create_sample_data(output_dir: str = "data"):
    """サンプルデータを作成"""
    logger = logging.getLogger(__name__)
    logger.info("サンプルデータを作成します")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # サンプルMarkdownファイル
    sample_md_content = """# LLM Complete Toolkit サンプル

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
"""
    
    # サンプルインストラクションデータ
    sample_instructions = [
        {
            "instruction": "Pythonでリストを作成する方法を教えてください。",
            "input": "",
            "output": "Pythonでリストを作成するには、角括弧[]を使用します。例: my_list = [1, 2, 3, 'hello']"
        },
        {
            "instruction": "for文の使い方を説明してください。",
            "input": "",
            "output": "for文は繰り返し処理を行うための構文です。例: for i in range(5): print(i)"
        },
        {
            "instruction": "機械学習における転移学習とは何ですか？",
            "input": "",
            "output": "転移学習は、事前学習済みモデルを新しいタスクに適応させる手法です。計算資源を節約し、少ないデータでも良い性能を得られます。"
        },
        {
            "instruction": "強化学習のPPOアルゴリズムについて説明してください。",
            "input": "",
            "output": "PPO（Proximal Policy Optimization）は、ポリシー勾配法の改良版で、学習の安定性を向上させた強化学習アルゴリズムです。"
        },
        {
            "instruction": "LLMのファインチューニングにLoRAを使う利点は何ですか？",
            "input": "",
            "output": "LoRAは低ランク適応により、全パラメータを更新することなく効率的にファインチューニングできます。メモリ使用量を大幅に削減できます。"
        }
    ]
    
    # ファイル作成
    sample_md_file = output_path / "sample_document.md"
    with open(sample_md_file, 'w', encoding='utf-8') as f:
        f.write(sample_md_content)
    
    train_file = output_path / "train.jsonl"
    with jsonlines.open(train_file, mode='w') as writer:
        for item in sample_instructions:
            writer.write(item)
    
    eval_file = output_path / "eval.jsonl"
    with jsonlines.open(eval_file, mode='w') as writer:
        for item in sample_instructions[:2]:
            writer.write(item)
    
    logger.info(f"サンプルデータを作成しました:")
    logger.info(f"  Markdownファイル: {sample_md_file}")
    logger.info(f"  訓練データ: {train_file}")
    logger.info(f"  評価データ: {eval_file}")


def main():
    setup_project_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description='LLM Complete Toolkit - 統合ツールキット',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ドキュメント抽出
  python main.py extract documents/ outputs/ --format jsonl
  
  # LoRAファインチューニング
  python main.py train-lora --train-data data/train.jsonl
  
  # QLoRAファインチューニング
  python main.py train-qlora --train-data data/train.jsonl
  
  # PPO強化学習
  python main.py train-rl --algorithm ppo
  
  # DQN強化学習
  python main.py train-rl --algorithm dqn
  
  # サンプルデータ作成
  python main.py create-samples
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # ドキュメント抽出サブコマンド
    extract_parser = subparsers.add_parser('extract', help='PDF/Markdownからデータ抽出')
    extract_parser.add_argument('input_dir', type=str, help='入力ディレクトリパス')
    extract_parser.add_argument('output_dir', type=str, help='出力ディレクトリパス')
    extract_parser.add_argument('--format', choices=['jsonl', 'text'], default='jsonl', 
                               help='出力フォーマット (default: jsonl)')
    extract_parser.add_argument('--chunk-size', type=int, default=512, 
                               help='テキストチャンクサイズ (default: 512)')
    extract_parser.add_argument('--instruction-format', action='store_true',
                               help='インストラクション形式でも出力')
    
    # LoRAサブコマンド
    lora_parser = subparsers.add_parser('train-lora', help='LoRAファインチューニング')
    lora_parser.add_argument('--train-data', type=str, required=True, help='訓練データファイル')
    lora_parser.add_argument('--eval-data', type=str, help='評価データファイル')
    lora_parser.add_argument('--config', type=str, default='configs/config.yaml', help='設定ファイル')
    lora_parser.add_argument('--output-dir', type=str, help='出力ディレクトリ')
    lora_parser.add_argument('--model-name', type=str, help='ベースモデル名')
    lora_parser.add_argument('--epochs', type=int, help='エポック数')
    lora_parser.add_argument('--batch-size', type=int, help='バッチサイズ')
    lora_parser.add_argument('--learning-rate', type=float, help='学習率')
    lora_parser.add_argument('--verbose', '-v', action='store_true', help='詳細ログ')
    
    # QLoRAサブコマンド
    qlora_parser = subparsers.add_parser('train-qlora', help='QLoRAファインチューニング')
    qlora_parser.add_argument('--train-data', type=str, required=True, help='訓練データファイル')
    qlora_parser.add_argument('--eval-data', type=str, help='評価データファイル')
    qlora_parser.add_argument('--config', type=str, default='configs/config.yaml', help='設定ファイル')
    qlora_parser.add_argument('--output-dir', type=str, help='出力ディレクトリ')
    qlora_parser.add_argument('--model-name', type=str, help='ベースモデル名')
    qlora_parser.add_argument('--epochs', type=int, help='エポック数')
    qlora_parser.add_argument('--batch-size', type=int, help='バッチサイズ')
    qlora_parser.add_argument('--learning-rate', type=float, help='学習率')
    qlora_parser.add_argument('--verbose', '-v', action='store_true', help='詳細ログ')
    
    # 強化学習サブコマンド
    rl_parser = subparsers.add_parser('train-rl', help='強化学習トレーニング')
    rl_parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn'], required=True, help='アルゴリズム')
    rl_parser.add_argument('--config', type=str, default='configs/config.yaml', help='設定ファイル')
    rl_parser.add_argument('--output-dir', type=str, help='出力ディレクトリ')
    rl_parser.add_argument('--episodes', type=int, help='エピソード数')
    rl_parser.add_argument('--verbose', '-v', action='store_true', help='詳細ログ')
    
    # サンプルデータ作成サブコマンド
    samples_parser = subparsers.add_parser('create-samples', help='サンプルデータ作成')
    samples_parser.add_argument('--output-dir', type=str, default='data', help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'extract':
            extract_documents(args)
        elif args.command == 'train-lora':
            launch_lora_training(args)
        elif args.command == 'train-qlora':
            launch_qlora_training(args)
        elif args.command == 'train-rl':
            launch_rl_training(args)
        elif args.command == 'create-samples':
            create_sample_data(args.output_dir)
        
        logger.info("処理が完了しました！")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()