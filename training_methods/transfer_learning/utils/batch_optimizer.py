"""
動的バッチサイズ最適化ユーティリティ
"""

import torch
import psutil
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BatchOptimizationConfig:
    """バッチ最適化設定"""
    initial_batch_size: int = 4
    max_batch_size: int = 16
    min_batch_size: int = 1
    memory_threshold: float = 0.85  # メモリ使用率閾値
    growth_factor: float = 1.5  # バッチサイズ増加率
    shrink_factor: float = 0.75  # バッチサイズ減少率
    gradient_accumulation_steps: int = 4
    max_gradient_accumulation: int = 16


class BatchOptimizer:
    """動的バッチサイズ最適化クラス"""
    
    def __init__(self, config: BatchOptimizationConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.current_gradient_accumulation = config.gradient_accumulation_steps
        self.logger = logging.getLogger(__name__)
        
        # メモリ監視
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        
    def get_memory_usage(self) -> Dict[str, float]:
        """メモリ使用状況を取得"""
        memory_info = {}
        
        # システムメモリ
        system_memory = psutil.virtual_memory()
        memory_info["system_used"] = system_memory.percent / 100.0
        memory_info["system_available"] = system_memory.available / (1024**3)  # GB
        
        # GPU/MPSメモリ
        if self.device.type == "cuda":
            memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
            memory_info["gpu_max"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            memory_info["gpu_usage"] = memory_info["gpu_allocated"] / memory_info["gpu_max"]
        elif self.device.type == "mps":
            memory_info["mps_allocated"] = torch.mps.current_allocated_memory() / (1024**3)  # GB
            memory_info["mps_usage"] = memory_info["system_used"]  # MPSはシステムメモリを使用
        
        return memory_info
    
    def optimize_batch_size(self, current_loss: Optional[float] = None) -> Tuple[int, int]:
        """
        バッチサイズとGradient Accumulationを最適化
        
        Args:
            current_loss: 現在の損失値
            
        Returns:
            (最適化されたバッチサイズ, Gradient Accumulation Steps)
        """
        memory_info = self.get_memory_usage()
        
        # メモリ使用率に基づく調整
        memory_usage = memory_info.get("gpu_usage", memory_info.get("mps_usage", memory_info["system_used"]))
        
        if memory_usage > self.config.memory_threshold:
            # メモリ使用率が高い場合はバッチサイズを減少
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * self.config.shrink_factor)
            )
            
            # バッチサイズを減らした分、Gradient Accumulationを増やす
            if new_batch_size < self.current_batch_size:
                ratio = self.current_batch_size / new_batch_size
                new_gradient_accumulation = min(
                    self.config.max_gradient_accumulation,
                    int(self.current_gradient_accumulation * ratio)
                )
            else:
                new_gradient_accumulation = self.current_gradient_accumulation
                
            self.logger.warning(f"メモリ使用率が高いため調整: {memory_usage:.2%} -> batch_size: {self.current_batch_size} -> {new_batch_size}")
            
        elif memory_usage < self.config.memory_threshold * 0.7:
            # メモリに余裕がある場合はバッチサイズを増加
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.growth_factor)
            )
            
            # バッチサイズを増やした分、Gradient Accumulationを減らす
            if new_batch_size > self.current_batch_size:
                ratio = self.current_batch_size / new_batch_size
                new_gradient_accumulation = max(
                    1,
                    int(self.current_gradient_accumulation * ratio)
                )
            else:
                new_gradient_accumulation = self.current_gradient_accumulation
                
            self.logger.info(f"メモリに余裕があるため調整: {memory_usage:.2%} -> batch_size: {self.current_batch_size} -> {new_batch_size}")
            
        else:
            # 現状維持
            new_batch_size = self.current_batch_size
            new_gradient_accumulation = self.current_gradient_accumulation
        
        # 更新
        self.current_batch_size = new_batch_size
        self.current_gradient_accumulation = new_gradient_accumulation
        
        return new_batch_size, new_gradient_accumulation
    
    def get_effective_batch_size(self) -> int:
        """実効バッチサイズを計算"""
        return self.current_batch_size * self.current_gradient_accumulation
    
    def log_memory_stats(self):
        """メモリ統計をログ出力"""
        memory_info = self.get_memory_usage()
        
        self.logger.info("=== メモリ使用状況 ===")
        self.logger.info(f"システムメモリ使用率: {memory_info['system_used']:.2%}")
        self.logger.info(f"システムメモリ空き: {memory_info['system_available']:.2f} GB")
        
        if self.device.type == "cuda":
            self.logger.info(f"GPU使用率: {memory_info['gpu_usage']:.2%}")
            self.logger.info(f"GPU割り当て: {memory_info['gpu_allocated']:.2f} GB")
            self.logger.info(f"GPU予約済み: {memory_info['gpu_reserved']:.2f} GB")
            self.logger.info(f"GPU最大: {memory_info['gpu_max']:.2f} GB")
        elif self.device.type == "mps":
            self.logger.info(f"MPS割り当て: {memory_info['mps_allocated']:.2f} GB")
            self.logger.info(f"MPS使用率: {memory_info['mps_usage']:.2%}")
        
        self.logger.info(f"現在のバッチサイズ: {self.current_batch_size}")
        self.logger.info(f"Gradient Accumulation: {self.current_gradient_accumulation}")
        self.logger.info(f"実効バッチサイズ: {self.get_effective_batch_size()}")
        self.logger.info("=" * 30)
    
    def handle_oom_error(self) -> Tuple[int, int]:
        """
        OOMエラー処理
        
        Returns:
            (調整後のバッチサイズ, Gradient Accumulation Steps)
        """
        self.logger.error("OOMエラーが発生しました。バッチサイズを調整します。")
        
        # バッチサイズを半分に
        new_batch_size = max(self.config.min_batch_size, self.current_batch_size // 2)
        
        # Gradient Accumulationを調整
        if new_batch_size < self.current_batch_size:
            ratio = self.current_batch_size / new_batch_size
            new_gradient_accumulation = min(
                self.config.max_gradient_accumulation,
                int(self.current_gradient_accumulation * ratio)
            )
        else:
            new_gradient_accumulation = self.current_gradient_accumulation
        
        self.current_batch_size = new_batch_size
        self.current_gradient_accumulation = new_gradient_accumulation
        
        self.logger.info(f"調整後: batch_size={new_batch_size}, gradient_accumulation={new_gradient_accumulation}")
        
        return new_batch_size, new_gradient_accumulation


class AdaptiveTrainingCallback:
    """適応的学習用コールバック"""
    
    def __init__(self, batch_optimizer: BatchOptimizer, adjustment_frequency: int = 100):
        self.batch_optimizer = batch_optimizer
        self.adjustment_frequency = adjustment_frequency
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
    
    def on_step_begin(self, trainer, step: int):
        """ステップ開始時の処理"""
        self.step_count += 1
        
        if self.step_count % self.adjustment_frequency == 0:
            # 定期的にバッチサイズを最適化
            new_batch_size, new_gradient_accumulation = self.batch_optimizer.optimize_batch_size()
            
            # TrainingArgumentsを更新
            if (new_batch_size != trainer.args.per_device_train_batch_size or 
                new_gradient_accumulation != trainer.args.gradient_accumulation_steps):
                
                trainer.args.per_device_train_batch_size = new_batch_size
                trainer.args.gradient_accumulation_steps = new_gradient_accumulation
                
                self.logger.info(f"Step {step}: バッチサイズ更新 -> {new_batch_size}, GA: {new_gradient_accumulation}")
    
    def on_step_end(self, trainer, step: int):
        """ステップ終了時の処理"""
        if self.step_count % (self.adjustment_frequency * 5) == 0:
            # 定期的にメモリ統計を出力
            self.batch_optimizer.log_memory_stats()