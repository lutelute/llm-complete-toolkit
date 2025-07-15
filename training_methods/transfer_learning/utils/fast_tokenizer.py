"""
高速トークナイザー（安定性重視）
"""

import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class FastTokenizer:
    """高速トークナイザー（キャッシュ機能付き）"""
    
    def __init__(self, tokenizer: AutoTokenizer, cache_dir: str = "./cache"):
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
    def _get_cache_key(self, texts: List[str], max_length: int) -> str:
        """キャッシュキーを生成"""
        content = str(texts[:10]) + str(max_length) + str(len(texts))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """キャッシュファイルパスを取得"""
        return self.cache_dir / f"tokenized_{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dataset]:
        """キャッシュから読み込み"""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.logger.info(f"キャッシュから読み込み: {cache_path}")
                    return Dataset.from_dict(data)
            except Exception as e:
                self.logger.warning(f"キャッシュ読み込みエラー: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def _save_to_cache(self, cache_key: str, dataset: Dataset):
        """キャッシュに保存"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with self._lock:
                with open(cache_path, 'wb') as f:
                    pickle.dump(dict(dataset), f)
                self.logger.info(f"キャッシュに保存: {cache_path}")
        except Exception as e:
            self.logger.error(f"キャッシュ保存エラー: {e}")
    
    def _tokenize_batch(self, texts: List[str], max_length: int) -> Dict[str, List]:
        """バッチトークナイゼーション"""
        try:
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # 教師ありファインチューニング用にlabelsを設定
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            # 長さ情報を追加
            tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
            
            return tokenized
            
        except Exception as e:
            self.logger.error(f"トークナイゼーションエラー: {e}")
            raise
    
    def _chunk_texts(self, texts: List[str], chunk_size: int = 1000) -> List[List[str]]:
        """テキストをチャンクに分割"""
        return [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    def tokenize_fast(self, texts: List[str], max_length: int = 512, 
                     use_cache: bool = True, chunk_size: int = 1000,
                     max_workers: int = 2) -> Dataset:
        """
        高速トークナイゼーション
        
        Args:
            texts: テキストのリスト
            max_length: 最大長
            use_cache: キャッシュを使用するか
            chunk_size: チャンクサイズ
            max_workers: 並列ワーカー数
            
        Returns:
            トークナイズされたデータセット
        """
        if use_cache:
            cache_key = self._get_cache_key(texts, max_length)
            cached_dataset = self._load_from_cache(cache_key)
            if cached_dataset:
                return cached_dataset
        
        self.logger.info(f"高速トークナイゼーション開始: {len(texts)}件")
        
        # テキストをチャンクに分割
        text_chunks = self._chunk_texts(texts, chunk_size)
        
        all_results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "length": []
        }
        
        if max_workers == 1:
            # シングルスレッド処理（安定性優先）
            for i, chunk in enumerate(text_chunks):
                self.logger.info(f"処理中: チャンク {i+1}/{len(text_chunks)}")
                result = self._tokenize_batch(chunk, max_length)
                
                for key in all_results.keys():
                    all_results[key].extend(result[key])
        else:
            # マルチスレッド処理（速度優先）
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for i, chunk in enumerate(text_chunks):
                    future = executor.submit(self._tokenize_batch, chunk, max_length)
                    futures.append((i, future))
                
                for i, future in futures:
                    try:
                        result = future.result()
                        for key in all_results.keys():
                            all_results[key].extend(result[key])
                        
                        self.logger.info(f"完了: チャンク {i+1}/{len(text_chunks)}")
                        
                    except Exception as e:
                        self.logger.error(f"チャンク {i+1} でエラー: {e}")
                        raise
        
        # データセットを作成
        dataset = Dataset.from_dict(all_results)
        
        # キャッシュに保存
        if use_cache:
            self._save_to_cache(cache_key, dataset)
        
        self.logger.info(f"高速トークナイゼーション完了: {len(dataset)}件")
        return dataset
    
    def clear_cache(self):
        """キャッシュをクリア"""
        try:
            for cache_file in self.cache_dir.glob("tokenized_*.pkl"):
                cache_file.unlink()
            self.logger.info("キャッシュをクリアしました")
        except Exception as e:
            self.logger.error(f"キャッシュクリアエラー: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        cache_files = list(self.cache_dir.glob("tokenized_*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


class OptimizedDataProcessor:
    """最適化されたデータ処理クラス"""
    
    def __init__(self, tokenizer: AutoTokenizer, cache_dir: str = "./cache"):
        self.tokenizer = tokenizer
        self.fast_tokenizer = FastTokenizer(tokenizer, cache_dir)
        self.logger = logging.getLogger(__name__)
    
    def prepare_dataset_optimized(self, texts: List[str], max_length: int = 512,
                                 use_cache: bool = True, chunk_size: int = 2000,
                                 max_workers: int = 1) -> Dataset:
        """
        最適化されたデータセット準備
        
        Args:
            texts: テキストのリスト
            max_length: 最大長
            use_cache: キャッシュを使用するか
            chunk_size: チャンクサイズ
            max_workers: 並列ワーカー数（1で安定性優先）
            
        Returns:
            最適化されたデータセット
        """
        self.logger.info(f"最適化データ処理開始: {len(texts)}件")
        
        # 高速トークナイゼーション
        dataset = self.fast_tokenizer.tokenize_fast(
            texts=texts,
            max_length=max_length,
            use_cache=use_cache,
            chunk_size=chunk_size,
            max_workers=max_workers
        )
        
        self.logger.info(f"最適化データ処理完了: {len(dataset)}件")
        return dataset
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        テキストの前処理
        
        Args:
            texts: 元のテキスト
            
        Returns:
            前処理されたテキスト
        """
        processed_texts = []
        
        for text in texts:
            # 基本的なクリーニング
            text = text.strip()
            
            # 空のテキストをスキップ
            if not text:
                continue
            
            # 長すぎるテキストを制限
            if len(text) > 10000:
                text = text[:10000]
            
            processed_texts.append(text)
        
        return processed_texts
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        cache_info = self.fast_tokenizer.get_cache_info()
        
        return {
            "cache_info": cache_info,
            "tokenizer_model": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size
        }