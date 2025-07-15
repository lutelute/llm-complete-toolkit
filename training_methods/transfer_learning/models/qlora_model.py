"""
QLoRA (Quantized Low-Rank Adaptation) モデルの実装
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from typing import Dict, List, Optional, Union
import logging
import os


class QLoRAFineTuner:
    def __init__(
        self,
        model_name: str,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        task_type: TaskType = TaskType.CAUSAL_LM,
        load_in_4bit: bool = True,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        QLoRAファインチューニングクラス
        
        Args:
            model_name: ベースモデル名
            lora_r: LoRAのランク
            lora_alpha: LoRAのアルファ値
            lora_dropout: ドロップアウト率
            target_modules: LoRAを適用するモジュール
            task_type: タスクの種類
            load_in_4bit: 4bit量子化を使用するかどうか
            bnb_4bit_use_double_quant: ダブル量子化を使用するかどうか
            bnb_4bit_quant_type: 量子化タイプ
            bnb_4bit_compute_dtype: 計算データタイプ
            device: デバイス
        """
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # 量子化設定
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
        )
        
        # トークナイザーの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ベースモデルの読み込み（量子化あり）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # モデルをkbitトレーニング用に準備
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
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
    
    def prepare_dataset(self, texts: List[str], max_length: int = 512):
        """
        データセットの準備
        
        Args:
            texts: テキストのリスト
            max_length: 最大長
            
        Returns:
            トークナイズされたデータセット
        """
        def tokenize_function(examples):
            # テキストをトークナイズ
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 教師ありファインチューニング用にlabelsを設定
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # データセットの作成
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def prepare_instruction_dataset(self, instructions: List[Dict[str, str]], max_length: int = 512):
        """
        インストラクションデータセットの準備
        
        Args:
            instructions: インストラクションデータ [{"instruction": str, "input": str, "output": str}]
            max_length: 最大長
            
        Returns:
            トークナイズされたデータセット
        """
        def format_instruction(instruction, input_text="", output=""):
            if input_text:
                return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        formatted_texts = []
        for item in instructions:
            formatted_text = format_instruction(
                item["instruction"],
                item.get("input", ""),
                item["output"]
            )
            formatted_texts.append(formatted_text)
        
        return self.prepare_dataset(formatted_texts, max_length)
    
    def train(
        self,
        train_dataset,
        output_dir: str = "./qlora_output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        warmup_steps: int = 100,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_dataset=None,
        **kwargs
    ):
        """
        QLoRAファインチューニングの実行
        
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
        
        # トレーニング引数の設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            bf16=True,  # QLoRAではbf16を推奨
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="tensorboard",
            remove_unused_columns=False,
            **kwargs
        )
        
        # データコレーターの設定
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # トレーナーの作成
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # トレーニング実行
        self.logger.info("QLoRAファインチューニング開始")
        trainer.train()
        
        # モデルの保存
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"QLoRAファインチューニング完了: {output_dir}")
        
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
    
    def save_qlora_weights(self, path: str):
        """QLoRAの重みのみを保存"""
        self.model.save_pretrained(path)
        self.logger.info(f"QLoRA重みを保存しました: {path}")
    
    def load_qlora_weights(self, path: str):
        """QLoRAの重みを読み込み"""
        self.model = PeftModel.from_pretrained(self.base_model, path)
        self.logger.info(f"QLoRA重みを読み込みました: {path}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量を取得"""
        if torch.cuda.is_available():
            return {
                "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
                "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9
            }
        return {"memory_usage": "CUDA not available"}
    
    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """モデルサイズ情報を取得"""
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = param_count * 4 / 1e6  # float32として計算
        
        return {
            "total_parameters": param_count,
            "estimated_size_mb": param_size_mb,
            "estimated_size_gb": param_size_mb / 1000
        }