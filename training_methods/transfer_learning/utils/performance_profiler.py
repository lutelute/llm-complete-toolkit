"""
学習パフォーマンスプロファイラー
"""

import time
import psutil
import torch
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from collections import defaultdict, deque


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    timestamp: float
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # 時間メトリクス
    step_time: Optional[float] = None
    forward_time: Optional[float] = None
    backward_time: Optional[float] = None
    optimizer_time: Optional[float] = None
    
    # メモリメトリクス
    system_memory_usage: Optional[float] = None
    system_memory_available: Optional[float] = None
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_reserved: Optional[float] = None
    mps_memory_allocated: Optional[float] = None
    
    # バッチメトリクス
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    effective_batch_size: Optional[int] = None
    
    # CPU/GPU使用率
    cpu_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    
    # スループット
    samples_per_second: Optional[float] = None
    tokens_per_second: Optional[float] = None


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""
    
    def __init__(self, output_dir: str = "./profiler_output", monitoring_interval: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(__name__)
        
        # デバイス情報
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        
        # モニタリング用変数
        self.is_monitoring = False
        self.monitoring_thread = None
        self.current_step = 0
        self.step_start_time = None
        self.phase_times = {}
        
        # 統計用
        self.step_times = deque(maxlen=100)  # 最近100ステップの時間
        self.memory_usage_history = deque(maxlen=100)
        
        self.logger.info(f"パフォーマンスプロファイラー初期化完了 (デバイス: {self.device})")
    
    def start_monitoring(self):
        """バックグラウンドモニタリング開始"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("バックグラウンドモニタリング開始")
    
    def stop_monitoring(self):
        """バックグラウンドモニタリング停止"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join()
            self.logger.info("バックグラウンドモニタリング停止")
    
    def _monitor_loop(self):
        """バックグラウンドモニタリングループ"""
        while self.is_monitoring:
            try:
                # システムメトリクスを収集
                memory_info = self._get_memory_info()
                cpu_usage = psutil.cpu_percent(interval=None)
                
                # メモリ使用履歴を更新
                self.memory_usage_history.append(memory_info)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"モニタリングエラー: {e}")
                time.sleep(self.monitoring_interval)
    
    def _get_memory_info(self) -> Dict[str, float]:
        """メモリ情報を取得"""
        memory_info = {}
        
        # システムメモリ
        system_memory = psutil.virtual_memory()
        memory_info["system_usage"] = system_memory.percent / 100.0
        memory_info["system_available"] = system_memory.available / (1024**3)  # GB
        
        # GPU/MPSメモリ
        if self.device.type == "cuda":
            memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
            memory_info["gpu_max"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        elif self.device.type == "mps":
            memory_info["mps_allocated"] = torch.mps.current_allocated_memory() / (1024**3)  # GB
        
        return memory_info
    
    @contextmanager
    def profile_step(self, step: int, batch_size: int = None):
        """ステップのプロファイリング"""
        self.current_step = step
        self.step_start_time = time.time()
        self.phase_times = {}
        
        try:
            yield self
        finally:
            step_end_time = time.time()
            step_time = step_end_time - self.step_start_time
            self.step_times.append(step_time)
            
            # メトリクスを記録
            self._record_step_metrics(step, step_time, batch_size)
    
    @contextmanager
    def profile_phase(self, phase_name: str):
        """フェーズのプロファイリング"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.phase_times[phase_name] = end_time - start_time
    
    def _record_step_metrics(self, step: int, step_time: float, batch_size: Optional[int] = None):
        """ステップメトリクスを記録"""
        memory_info = self._get_memory_info()
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            step=step,
            step_time=step_time,
            forward_time=self.phase_times.get("forward"),
            backward_time=self.phase_times.get("backward"),
            optimizer_time=self.phase_times.get("optimizer"),
            system_memory_usage=memory_info["system_usage"],
            system_memory_available=memory_info["system_available"],
            gpu_memory_allocated=memory_info.get("gpu_allocated"),
            gpu_memory_reserved=memory_info.get("gpu_reserved"),
            mps_memory_allocated=memory_info.get("mps_allocated"),
            batch_size=batch_size,
            cpu_usage=psutil.cpu_percent(interval=None),
            samples_per_second=batch_size / step_time if batch_size else None
        )
        
        self.metrics_history.append(metrics)
    
    def record_loss(self, loss: float, learning_rate: float = None):
        """損失値を記録"""
        if self.metrics_history:
            self.metrics_history[-1].loss = loss
            self.metrics_history[-1].learning_rate = learning_rate
    
    def record_batch_info(self, batch_size: int, gradient_accumulation_steps: int = None):
        """バッチ情報を記録"""
        if self.metrics_history:
            self.metrics_history[-1].batch_size = batch_size
            self.metrics_history[-1].gradient_accumulation_steps = gradient_accumulation_steps
            if gradient_accumulation_steps:
                self.metrics_history[-1].effective_batch_size = batch_size * gradient_accumulation_steps
    
    def get_current_stats(self) -> Dict[str, Any]:
        """現在の統計情報を取得"""
        if not self.step_times:
            return {}
        
        recent_times = list(self.step_times)
        recent_memory = list(self.memory_usage_history)
        
        stats = {
            "avg_step_time": sum(recent_times) / len(recent_times),
            "min_step_time": min(recent_times),
            "max_step_time": max(recent_times),
            "steps_per_second": 1.0 / (sum(recent_times) / len(recent_times)),
            "total_steps": len(self.metrics_history),
            "current_step": self.current_step
        }
        
        if recent_memory:
            latest_memory = recent_memory[-1]
            stats.update({
                "current_memory_usage": latest_memory["system_usage"],
                "available_memory_gb": latest_memory["system_available"]
            })
            
            if "gpu_allocated" in latest_memory:
                stats["gpu_memory_gb"] = latest_memory["gpu_allocated"]
            if "mps_allocated" in latest_memory:
                stats["mps_memory_gb"] = latest_memory["mps_allocated"]
        
        return stats
    
    def generate_report(self) -> Dict[str, Any]:
        """詳細レポートを生成"""
        if not self.metrics_history:
            return {"error": "メトリクスデータがありません"}
        
        # 基本統計
        step_times = [m.step_time for m in self.metrics_history if m.step_time]
        memory_usages = [m.system_memory_usage for m in self.metrics_history if m.system_memory_usage]
        
        report = {
            "summary": {
                "total_steps": len(self.metrics_history),
                "total_time": sum(step_times) if step_times else 0,
                "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
                "min_step_time": min(step_times) if step_times else 0,
                "max_step_time": max(step_times) if step_times else 0,
                "avg_memory_usage": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                "max_memory_usage": max(memory_usages) if memory_usages else 0,
            },
            "device_info": {
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "mps_available": torch.backends.mps.is_available(),
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """パフォーマンス改善推奨事項を生成"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        # メモリ使用量の分析
        memory_usages = [m.system_memory_usage for m in self.metrics_history if m.system_memory_usage]
        if memory_usages:
            avg_memory = sum(memory_usages) / len(memory_usages)
            max_memory = max(memory_usages)
            
            if max_memory > 0.9:
                recommendations.append("メモリ使用量が90%を超えています。バッチサイズを減らすことを検討してください。")
            elif avg_memory < 0.6:
                recommendations.append("メモリ使用量が60%未満です。バッチサイズを増やして効率化できる可能性があります。")
        
        # ステップ時間の分析
        step_times = [m.step_time for m in self.metrics_history if m.step_time]
        if len(step_times) > 10:
            recent_times = step_times[-10:]
            old_times = step_times[:10]
            
            if sum(recent_times) / len(recent_times) > sum(old_times) / len(old_times) * 1.2:
                recommendations.append("最近のステップ時間が増加しています。メモリリークやバッチサイズの調整を確認してください。")
        
        # デバイス固有の推奨事項
        if self.device.type == "mps":
            recommendations.append("Apple Silicon (MPS) を使用中です。bf16精度を有効にすると性能が向上する可能性があります。")
        elif self.device.type == "cpu":
            recommendations.append("CPU学習を実行中です。可能であればGPU環境での実行を検討してください。")
        
        return recommendations
    
    def save_metrics(self, filename: str = "performance_metrics.json"):
        """メトリクスをファイルに保存"""
        filepath = self.output_dir / filename
        
        # メトリクスを辞書形式に変換
        metrics_data = [asdict(metric) for metric in self.metrics_history]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"メトリクスを保存しました: {filepath}")
    
    def save_report(self, filename: str = "performance_report.json"):
        """レポートをファイルに保存"""
        filepath = self.output_dir / filename
        report = self.generate_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"レポートを保存しました: {filepath}")
    
    def print_summary(self):
        """サマリーを出力"""
        stats = self.get_current_stats()
        
        print("=" * 50)
        print("パフォーマンスサマリー")
        print("=" * 50)
        
        if stats:
            print(f"総ステップ数: {stats.get('total_steps', 'N/A')}")
            print(f"現在のステップ: {stats.get('current_step', 'N/A')}")
            print(f"平均ステップ時間: {stats.get('avg_step_time', 0):.3f}秒")
            print(f"ステップ/秒: {stats.get('steps_per_second', 0):.2f}")
            print(f"メモリ使用率: {stats.get('current_memory_usage', 0):.1%}")
            
            if "gpu_memory_gb" in stats:
                print(f"GPU メモリ使用量: {stats['gpu_memory_gb']:.2f} GB")
            if "mps_memory_gb" in stats:
                print(f"MPS メモリ使用量: {stats['mps_memory_gb']:.2f} GB")
        
        print("=" * 50)
    
    def cleanup(self):
        """クリーンアップ"""
        self.stop_monitoring()
        self.save_metrics()
        self.save_report()
        self.logger.info("プロファイラーをクリーンアップしました")


class TrainingProfilerCallback:
    """学習用プロファイラーコールバック"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.current_step = 0
        self.step_context = None
    
    def on_train_begin(self, trainer):
        """学習開始時の処理"""
        self.profiler.start_monitoring()
        self.profiler.logger.info("学習プロファイリング開始")
    
    def on_step_begin(self, trainer, step: int):
        """ステップ開始時の処理"""
        self.current_step = step
        batch_size = trainer.args.per_device_train_batch_size
        self.step_context = self.profiler.profile_step(step, batch_size)
        self.step_context.__enter__()
    
    def on_step_end(self, trainer, step: int):
        """ステップ終了時の処理"""
        if self.step_context:
            self.step_context.__exit__(None, None, None)
            self.step_context = None
        
        # バッチ情報を記録
        self.profiler.record_batch_info(
            trainer.args.per_device_train_batch_size,
            trainer.args.gradient_accumulation_steps
        )
        
        # 定期的に統計を出力
        if step % 100 == 0:
            self.profiler.print_summary()
    
    def on_train_end(self, trainer):
        """学習終了時の処理"""
        self.profiler.cleanup()
        self.profiler.logger.info("学習プロファイリング終了")