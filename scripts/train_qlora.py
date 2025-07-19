#!/usr/bin/env python3
"""
QLoRAファインチューニングスクリプト
"""

import argparse
import yaml
import sys
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_methods.transfer_learning.models.qlora_model import QLoRAFineTuner
from shared_utils.data_loader import DataLoaderFactory, DataValidator
from shared_utils.training_utils import setup_logging, set_seed, MetricsLogger


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='QLoRAファインチューニング')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='設定ファイルパス')
    parser.add_argument('--train-data', type=str, required=True,
                       help='訓練データファイルパス')
    parser.add_argument('--eval-data', type=str, 
                       help='評価データファイルパス')
    parser.add_argument('--output-dir', type=str, default='./outputs/qlora',
                       help='出力ディレクトリ')
    parser.add_argument('--model-name', type=str,
                       help='ベースモデル名（設定ファイルを上書き）')
    parser.add_argument('--epochs', type=int,
                       help='エポック数（設定ファイルを上書き）')
    parser.add_argument('--batch-size', type=int,
                       help='バッチサイズ（設定ファイルを上書き）')
    parser.add_argument('--learning-rate', type=float,
                       help='学習率（設定ファイルを上書き）')
    parser.add_argument('--max-length', type=int,
                       help='最大トークン長（設定ファイルを上書き）')
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細ログ表示')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config(args.config)
    qlora_config = config['transfer_learning']['qlora']
    common_config = config['common']
    
    # コマンドライン引数で設定を上書き
    if args.model_name:
        qlora_config['model_name'] = args.model_name
    if args.epochs:
        qlora_config['training']['num_train_epochs'] = args.epochs
    if args.batch_size:
        qlora_config['training']['per_device_train_batch_size'] = args.batch_size
    if args.learning_rate:
        qlora_config['training']['learning_rate'] = args.learning_rate
    if args.max_length:
        qlora_config['training']['max_length'] = args.max_length
    
    # ログ設定
    log_level = "DEBUG" if args.verbose else common_config.get('log_level', 'INFO')
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # シード設定
    set_seed(args.seed)
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"QLoRAファインチューニング開始")
    logger.info(f"ベースモデル: {qlora_config['model_name']}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    try:
        # データ読み込み
        logger.info(f"訓練データ読み込み: {args.train_data}")
        
        # インストラクションデータとして読み込み
        train_instructions = DataLoaderFactory.load_instruction_data(args.train_data)
        
        # データ検証
        validation_result = DataValidator.validate_instruction_data(train_instructions)
        if not validation_result['valid']:
            logger.error(f"訓練データの検証エラー: {validation_result['error']}")
            sys.exit(1)
        
        logger.info(f"訓練データ検証成功: {validation_result['count']}件")
        
        # 評価データ読み込み（オプション）
        eval_instructions = None
        if args.eval_data:
            logger.info(f"評価データ読み込み: {args.eval_data}")
            eval_instructions = DataLoaderFactory.load_instruction_data(args.eval_data)
            
            eval_validation = DataValidator.validate_instruction_data(eval_instructions)
            if not eval_validation['valid']:
                logger.error(f"評価データの検証エラー: {eval_validation['error']}")
                sys.exit(1)
            
            logger.info(f"評価データ検証成功: {eval_validation['count']}件")
        
        # QLoRAファインチューナーの初期化
        fine_tuner = QLoRAFineTuner(
            model_name=qlora_config['model_name'],
            lora_r=qlora_config['lora_r'],
            lora_alpha=qlora_config['lora_alpha'],
            lora_dropout=qlora_config['lora_dropout'],
            target_modules=qlora_config['target_modules'],
            **qlora_config['quantization']
        )
        
        # メモリ使用量の表示
        memory_info = fine_tuner.get_memory_usage()
        logger.info(f"メモリ使用量: {memory_info}")
        
        # モデルサイズの表示
        model_size = fine_tuner.get_model_size()
        logger.info(f"モデルサイズ: {model_size}")
        
        # データセット準備
        train_dataset = fine_tuner.prepare_instruction_dataset(
            train_instructions,
            max_length=qlora_config['training']['max_length']
        )
        
        eval_dataset = None
        if eval_instructions:
            eval_dataset = fine_tuner.prepare_instruction_dataset(
                eval_instructions,
                max_length=qlora_config['training']['max_length']
            )
        
        # トレーニング実行
        trainer = fine_tuner.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_dir),
            **qlora_config['training']
        )
        
        # テスト生成
        test_prompt = "### Instruction:\nPythonでリストをソートする方法を教えてください。\n\n### Response:\n"
        generated_text = fine_tuner.generate(test_prompt, max_length=150)
        logger.info(f"テスト生成結果: {generated_text}")
        
        # メモリ使用量の最終確認
        final_memory = fine_tuner.get_memory_usage()
        logger.info(f"最終メモリ使用量: {final_memory}")
        
        # 設定ファイルの保存
        config_save_path = output_dir / "training_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"QLoRAファインチューニング完了: {output_dir}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()