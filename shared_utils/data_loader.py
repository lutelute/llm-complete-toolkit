"""
データローダーとデータ処理ユーティリティ
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import jsonlines
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, load_dataset


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        return_tensors: str = "pt"
    ):
        """
        テキストデータセット
        
        Args:
            texts: テキストのリスト
            tokenizer: トークナイザー
            max_length: 最大長
            return_tensors: 返すテンソルの形式
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class InstructionDataset(Dataset):
    def __init__(
        self,
        instructions: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ):
        """
        インストラクションデータセット
        
        Args:
            instructions: インストラクションデータ
            tokenizer: トークナイザー
            max_length: 最大長
            instruction_template: インストラクションテンプレート
        """
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        item = self.instructions[idx]
        
        # インストラクション形式にフォーマット
        formatted_text = self.instruction_template.format(
            instruction=item["instruction"],
            input=item.get("input", ""),
            output=item["output"]
        )
        
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class RLDataset(Dataset):
    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ):
        """
        強化学習用データセット
        
        Args:
            states: 状態のリスト
            actions: アクションのリスト
            rewards: 報酬のリスト
            next_states: 次の状態のリスト
            dones: 終了フラグのリスト
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            "state": torch.FloatTensor(self.states[idx]),
            "action": torch.LongTensor([self.actions[idx]]),
            "reward": torch.FloatTensor([self.rewards[idx]]),
            "next_state": torch.FloatTensor(self.next_states[idx]),
            "done": torch.BoolTensor([self.dones[idx]])
        }


class DataLoaderFactory:
    """データローダーファクトリークラス"""
    
    @staticmethod
    def load_text_data(file_path: Union[str, Path]) -> List[str]:
        """
        テキストファイルからデータを読み込み
        
        Args:
            file_path: ファイルパス
            
        Returns:
            テキストのリスト
        """
        file_path = Path(file_path)
        
        if file_path.suffix == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        
        elif file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(item) for item in data]
                else:
                    return [str(data)]
        
        elif file_path.suffix == ".jsonl":
            texts = []
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    if isinstance(item, dict):
                        texts.append(item.get("text", str(item)))
                    else:
                        texts.append(str(item))
            return texts
        
        elif file_path.suffix in [".csv", ".tsv"]:
            separator = "\t" if file_path.suffix == ".tsv" else ","
            df = pd.read_csv(file_path, sep=separator)
            if "text" in df.columns:
                return df["text"].tolist()
            else:
                return df.iloc[:, 0].astype(str).tolist()
        
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_path.suffix}")
    
    @staticmethod
    def load_instruction_data(file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        インストラクションデータを読み込み
        
        Args:
            file_path: ファイルパス
            
        Returns:
            インストラクションデータのリスト
        """
        file_path = Path(file_path)
        
        if file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif file_path.suffix == ".jsonl":
            instructions = []
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    instructions.append(item)
            return instructions
        
        else:
            raise ValueError(f"インストラクションデータは .json または .jsonl ファイルである必要があります")
    
    @staticmethod
    def create_text_dataloader(
        texts: List[str],
        tokenizer: AutoTokenizer,
        batch_size: int = 16,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        テキストデータローダーを作成
        
        Args:
            texts: テキストのリスト
            tokenizer: トークナイザー
            batch_size: バッチサイズ
            max_length: 最大長
            shuffle: シャッフルするかどうか
            num_workers: ワーカー数
            
        Returns:
            データローダー
        """
        dataset = TextDataset(texts, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    @staticmethod
    def create_instruction_dataloader(
        instructions: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        batch_size: int = 16,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        インストラクションデータローダーを作成
        
        Args:
            instructions: インストラクションデータ
            tokenizer: トークナイザー
            batch_size: バッチサイズ
            max_length: 最大長
            shuffle: シャッフルするかどうか
            num_workers: ワーカー数
            
        Returns:
            データローダー
        """
        dataset = InstructionDataset(instructions, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    @staticmethod
    def create_huggingface_dataset(
        data: Union[List[str], List[Dict[str, str]]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        is_instruction: bool = False
    ) -> HFDataset:
        """
        Hugging Face形式のデータセットを作成
        
        Args:
            data: データ
            tokenizer: トークナイザー
            max_length: 最大長
            is_instruction: インストラクションデータかどうか
            
        Returns:
            Hugging Faceデータセット
        """
        if is_instruction:
            # インストラクション形式
            def format_instruction(item):
                return f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            
            texts = [format_instruction(item) for item in data]
        else:
            # プレーンテキスト
            texts = data
        
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        dataset = HFDataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset


class DataValidator:
    """データ検証ユーティリティ"""
    
    @staticmethod
    def validate_text_data(texts: List[str]) -> Dict[str, Any]:
        """
        テキストデータの検証
        
        Args:
            texts: テキストのリスト
            
        Returns:
            検証結果
        """
        if not texts:
            return {"valid": False, "error": "テキストデータが空です"}
        
        lengths = [len(text) for text in texts]
        
        return {
            "valid": True,
            "count": len(texts),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "empty_texts": sum(1 for text in texts if not text.strip())
        }
    
    @staticmethod
    def validate_instruction_data(instructions: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        インストラクションデータの検証
        
        Args:
            instructions: インストラクションデータ
            
        Returns:
            検証結果
        """
        if not instructions:
            return {"valid": False, "error": "インストラクションデータが空です"}
        
        required_keys = {"instruction", "output"}
        missing_keys = []
        
        for i, item in enumerate(instructions):
            if not isinstance(item, dict):
                return {"valid": False, "error": f"アイテム {i} が辞書ではありません"}
            
            item_keys = set(item.keys())
            if not required_keys.issubset(item_keys):
                missing_keys.append(i)
        
        if missing_keys:
            return {
                "valid": False,
                "error": f"必要なキー({required_keys})が不足しているアイテム: {missing_keys[:5]}"
            }
        
        return {
            "valid": True,
            "count": len(instructions),
            "has_input": sum(1 for item in instructions if "input" in item and item["input"]),
            "avg_instruction_length": sum(len(item["instruction"]) for item in instructions) / len(instructions),
            "avg_output_length": sum(len(item["output"]) for item in instructions) / len(instructions)
        }