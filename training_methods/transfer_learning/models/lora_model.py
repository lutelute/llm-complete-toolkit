"""
LoRA (Low-Rank Adaptation) モデルの実装
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
from typing import Dict, List, Optional, Union
import logging
import os
from ..utils.batch_optimizer import BatchOptimizer, BatchOptimizationConfig, AdaptiveTrainingCallback
from ..utils.performance_profiler import PerformanceProfiler, TrainingProfilerCallback
from ..utils.fast_tokenizer import FastTokenizer, OptimizedDataProcessor
from ..utils.memory_optimizer import MemoryOptimizer, MemoryConfig, OptimizedDataCollator, MemoryOptimizedTrainingCallback


class LoRAFineTuner:
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        task_type: TaskType = TaskType.CAUSAL_LM,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        LoRAファインチューニングクラス
        
        Args:
            model_name: ベースモデル名
            lora_r: LoRAのランク
            lora_alpha: LoRAのアルファ値
            lora_dropout: ドロップアウト率
            target_modules: LoRAを適用するモジュール
            task_type: タスクの種類
            device: デバイス
        """
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # トークナイザーの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ベースモデルの読み込み
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA設定
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=task_type
        )
        
        # LoRAモデルの作成
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, texts: List[str], max_length: int = 512, use_fast_tokenizer: bool = True):
        """
        データセットの準備（高速化対応）
        
        Args:
            texts: テキストのリスト
            max_length: 最大長
            use_fast_tokenizer: 高速トークナイザーを使用するか
            
        Returns:
            トークナイズされたデータセット
        """
        if use_fast_tokenizer:
            # 高速トークナイザーを使用
            self.logger.info("高速トークナイザーを使用してデータセットを準備中...")
            
            processor = OptimizedDataProcessor(
                tokenizer=self.tokenizer,
                cache_dir=os.path.join(os.getcwd(), "cache", "tokenizer")
            )
            
            # テキストの前処理
            processed_texts = processor.preprocess_texts(texts)
            
            # 高速トークナイゼーション
            tokenized_dataset = processor.prepare_dataset_optimized(
                texts=processed_texts,
                max_length=max_length,
                use_cache=True,
                chunk_size=2000,
                max_workers=1  # 安定性優先
            )
            
            return tokenized_dataset
        
        else:
            # 従来の方法（フォールバック）
            self.logger.info("従来の方法でデータセットを準備中...")
            
            def tokenize_function(examples):
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_tensors=None
                )
                
                tokenized["labels"] = tokenized["input_ids"].copy()
                tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
                
                return tokenized
            
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": texts})
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                num_proc=1,
                remove_columns=["text"]
            )
            
            return tokenized_dataset
    
    def train(
        self,
        train_dataset,
        output_dir: str = "./lora_output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_dataset=None,
        use_adaptive_batching: bool = True,
        use_performance_profiler: bool = True,
        use_memory_optimization: bool = True,
        **kwargs
    ):
        """
        LoRAファインチューニングの実行
        
        Args:
            train_dataset: 訓練データセット
            output_dir: 出力ディレクトリ
            num_train_epochs: エポック数
            per_device_train_batch_size: バッチサイズ
            gradient_accumulation_steps: 勾配蓄積ステップ
            warmup_steps: ウォームアップステップ
            learning_rate: 学習率
            logging_steps: ログ出力間隔
            save_steps: 保存間隔
            eval_dataset: 評価データセット
            **kwargs: その他の引数
        """
        
        # 不要なパラメータを除去
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'max_length', 'dataloader_num_workers', 'group_by_length', 
            'use_adaptive_batching', 'use_performance_profiler', 'bf16_enabled',
            'use_memory_optimization'
        ]}
        
        # 動的バッチサイズ最適化の設定
        batch_optimizer = None
        adaptive_callback = None
        
        if use_adaptive_batching:
            batch_config = BatchOptimizationConfig(
                initial_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_batch_size=min(16, per_device_train_batch_size * 4),
                min_batch_size=max(1, per_device_train_batch_size // 4)
            )
            batch_optimizer = BatchOptimizer(batch_config)
            adaptive_callback = AdaptiveTrainingCallback(batch_optimizer)
            
            # 初期メモリ状況を確認
            batch_optimizer.log_memory_stats()
            
            # 最適化されたバッチサイズを取得
            per_device_train_batch_size, gradient_accumulation_steps = batch_optimizer.optimize_batch_size()
        
        # パフォーマンスプロファイラーの設定
        profiler = None
        profiler_callback = None
        
        if use_performance_profiler:
            profiler = PerformanceProfiler(
                output_dir=os.path.join(output_dir, "profiler_output")
            )
            profiler_callback = TrainingProfilerCallback(profiler)
        
        # メモリ最適化の設定
        memory_optimizer = None
        memory_callback = None
        
        if use_memory_optimization:
            memory_config = MemoryConfig(
                enable_gradient_checkpointing=True,
                use_memory_efficient_attention=True,
                optimize_memory_usage=True,
                gc_frequency=100,
                memory_cleanup_threshold=0.85
            )
            memory_optimizer = MemoryOptimizer(memory_config)
            memory_callback = MemoryOptimizedTrainingCallback(memory_optimizer)
            
            # モデルのメモリ最適化
            self.model = memory_optimizer.optimize_model(self.model)
        
        # トレーニング引数の設定（最適化対応）
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available() and not torch.backends.mps.is_available(),
            bf16=False,  # Apple Silicon環境では現在サポートされていない
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="tensorboard",
            dataloader_num_workers=4,  # 並列データローダー
            dataloader_pin_memory=True,  # メモリ効率化
            group_by_length=True,  # 長さでグループ化
            length_column_name="length",  # 長さ列名
            remove_unused_columns=False,  # 未使用列を保持
            **filtered_kwargs
        )
        
        # データコレーターの設定（最適化対応）
        if use_memory_optimization and memory_optimizer:
            data_collator = OptimizedDataCollator(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
                memory_optimizer=memory_optimizer
            )
        else:
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        
        # コールバックの設定
        callbacks = []
        if adaptive_callback:
            callbacks.append(adaptive_callback)
        if profiler_callback:
            callbacks.append(profiler_callback)
        if memory_callback:
            callbacks.append(memory_callback)
        
        # トレーナーの作成
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        # トレーニング実行
        self.logger.info("LoRAファインチューニング開始")
        trainer.train()
        
        # モデルの保存
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"LoRAファインチューニング完了: {output_dir}")
        
        return trainer
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_length: 最大長
            temperature: 温度パラメータ
            do_sample: サンプリングするかどうか
            top_p: top-pサンプリング
            top_k: top-kサンプリング
            
        Returns:
            生成されたテキスト
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
    
    def save_lora_weights(self, path: str):
        """LoRAの重みのみを保存"""
        self.model.save_pretrained(path)
        self.logger.info(f"LoRA重みを保存しました: {path}")
    
    def load_lora_weights(self, path: str):
        """LoRAの重みを読み込み"""
        self.model = PeftModel.from_pretrained(self.base_model, path)
        self.logger.info(f"LoRA重みを読み込みました: {path}")
    
    def merge_and_save(self, output_path: str):
        """LoRAをベースモデルにマージして保存"""
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.logger.info(f"マージされたモデルを保存しました: {output_path}")
    
    def get_trainable_params(self) -> Dict[str, int]:
        """訓練可能なパラメータ数を取得"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": 100 * trainable_params / all_param
        }