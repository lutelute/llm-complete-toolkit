"""
メモリ最適化ユーティリティ
"""

import gc
import torch
import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
from transformers import DataCollatorForLanguageModeling


@dataclass
class MemoryConfig:
    """メモリ最適化設定"""
    enable_gradient_checkpointing: bool = True
    use_memory_efficient_attention: bool = True
    optimize_memory_usage: bool = True
    gc_frequency: int = 100  # ガベージコレクション頻度
    memory_cleanup_threshold: float = 0.85  # メモリクリーンアップ閾値


class MemoryOptimizer:
    """メモリ最適化クラス"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        モデルのメモリ最適化
        
        Args:
            model: 最適化するモデル
            
        Returns:
            最適化されたモデル
        """
        if self.config.enable_gradient_checkpointing:
            # グラデーションチェックポイントを有効化
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("グラデーションチェックポイントを有効化しました")
        
        # メモリ効率の最適化
        if self.config.optimize_memory_usage:
            # モデルを評価モードに設定（推論時のメモリ節約）
            model.eval()
            
            # 不要なパラメータの勾配を無効化
            for param in model.parameters():
                if not param.requires_grad:
                    param.grad = None
        
        self.logger.info("モデルメモリ最適化完了")
        return model
    
    def get_memory_usage(self) -> Dict[str, float]:
        """現在のメモリ使用量を取得"""
        memory_info = {}
        
        # システムメモリ
        system_memory = psutil.virtual_memory()
        memory_info["system_used_gb"] = system_memory.used / (1024**3)
        memory_info["system_total_gb"] = system_memory.total / (1024**3)
        memory_info["system_usage_percent"] = system_memory.percent
        
        # GPU/MPSメモリ
        if self.device.type == "cuda":
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            memory_info["gpu_max_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif self.device.type == "mps":
            memory_info["mps_allocated_gb"] = torch.mps.current_allocated_memory() / (1024**3)
        
        return memory_info
    
    def cleanup_memory(self, force: bool = False):
        """メモリクリーンアップ"""
        memory_info = self.get_memory_usage()
        memory_usage = memory_info.get("system_usage_percent", 0) / 100.0
        
        if force or memory_usage > self.config.memory_cleanup_threshold:
            self.logger.info(f"メモリクリーンアップ実行 (使用率: {memory_usage:.1%})")
            
            # Python ガベージコレクション
            gc.collect()
            
            # PyTorchキャッシュクリア
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            
            # メモリ使用量を再確認
            new_memory_info = self.get_memory_usage()
            new_usage = new_memory_info.get("system_usage_percent", 0) / 100.0
            
            self.logger.info(f"クリーンアップ後メモリ使用率: {new_usage:.1%}")
    
    def step_cleanup(self):
        """ステップごとのメモリクリーンアップ"""
        self.step_count += 1
        
        if self.step_count % self.config.gc_frequency == 0:
            self.cleanup_memory()
    
    @contextmanager
    def memory_context(self):
        """メモリ管理コンテキスト"""
        initial_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            # 処理後のメモリクリーンアップ
            self.cleanup_memory(force=True)
            
            final_memory = self.get_memory_usage()
            self.logger.info(f"メモリ使用量変化: "
                           f"{initial_memory.get('system_usage_percent', 0):.1f}% → "
                           f"{final_memory.get('system_usage_percent', 0):.1f}%")


class OptimizedDataCollator(DataCollatorForLanguageModeling):
    """メモリ最適化されたデータコレクター"""
    
    def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=8, 
                 memory_optimizer: Optional[MemoryOptimizer] = None):
        super().__init__(tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of)
        self.memory_optimizer = memory_optimizer
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        メモリ効率的なバッチ作成
        
        Args:
            features: バッチ内のサンプル
            
        Returns:
            最適化されたバッチ
        """
        # メモリクリーンアップ
        if self.memory_optimizer:
            self.memory_optimizer.step_cleanup()
        
        # 効率的なパディング
        batch_size = len(features)
        max_length = max(len(f["input_ids"]) for f in features)
        
        # 不要な長さのパディングを避ける
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // 
                         self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # 最適化されたバッチを作成
        batch = {}
        
        for key in features[0].keys():
            if key == "input_ids":
                batch[key] = torch.zeros(batch_size, max_length, dtype=torch.long)
                for i, f in enumerate(features):
                    seq_len = len(f[key])
                    batch[key][i, :seq_len] = torch.tensor(f[key])
                    
            elif key == "attention_mask":
                batch[key] = torch.zeros(batch_size, max_length, dtype=torch.long)
                for i, f in enumerate(features):
                    seq_len = len(f["input_ids"])
                    batch[key][i, :seq_len] = 1
                    
            elif key == "labels":
                batch[key] = torch.full((batch_size, max_length), -100, dtype=torch.long)
                for i, f in enumerate(features):
                    seq_len = len(f[key])
                    batch[key][i, :seq_len] = torch.tensor(f[key])
        
        return batch


class MemoryMonitor:
    """メモリ監視クラス"""
    
    def __init__(self, memory_optimizer: MemoryOptimizer, 
                 monitoring_interval: float = 10.0):
        self.memory_optimizer = memory_optimizer
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.logger = logging.getLogger(__name__)
        
        # メモリ使用量履歴
        self.memory_history = []
        self.max_history_size = 100
    
    def start_monitoring(self):
        """監視開始"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("メモリ監視を開始しました")
    
    def stop_monitoring(self):
        """監視停止"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            self.logger.info("メモリ監視を停止しました")
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                memory_info = self.memory_optimizer.get_memory_usage()
                
                # 履歴に追加
                self.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": memory_info
                })
                
                # 履歴サイズ制限
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                
                # 高メモリ使用量の警告
                usage_percent = memory_info.get("system_usage_percent", 0)
                if usage_percent > 90:
                    self.logger.warning(f"高メモリ使用量: {usage_percent:.1f}%")
                    self.memory_optimizer.cleanup_memory(force=True)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"メモリ監視エラー: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """メモリ統計を取得"""
        if not self.memory_history:
            return {"error": "履歴データがありません"}
        
        recent_usage = [h["memory_info"].get("system_usage_percent", 0) 
                       for h in self.memory_history[-10:]]
        
        return {
            "current_usage": recent_usage[-1] if recent_usage else 0,
            "average_usage": sum(recent_usage) / len(recent_usage) if recent_usage else 0,
            "max_usage": max(recent_usage) if recent_usage else 0,
            "min_usage": min(recent_usage) if recent_usage else 0,
            "history_size": len(self.memory_history)
        }


class MemoryOptimizedTrainingCallback:
    """メモリ最適化トレーニングコールバック"""
    
    def __init__(self, memory_optimizer: MemoryOptimizer):
        self.memory_optimizer = memory_optimizer
        self.memory_monitor = MemoryMonitor(memory_optimizer)
        self.logger = logging.getLogger(__name__)
    
    def on_init_end(self, args, state, control, **kwargs):
        """初期化終了時"""
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        """学習開始時"""
        self.memory_monitor.start_monitoring()
        self.logger.info("メモリ最適化トレーニング開始")
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """ステップ終了時"""
        self.memory_optimizer.step_cleanup()
        
        # 定期的なメモリ統計出力
        if state.global_step % 500 == 0:
            stats = self.memory_monitor.get_memory_stats()
            self.logger.info(f"メモリ統計 (Step {state.global_step}): "
                           f"現在 {stats.get('current_usage', 0):.1f}%, "
                           f"平均 {stats.get('average_usage', 0):.1f}%")
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """学習終了時"""
        self.memory_monitor.stop_monitoring()
        self.memory_optimizer.cleanup_memory(force=True)
        self.logger.info("メモリ最適化トレーニング終了")
        return control