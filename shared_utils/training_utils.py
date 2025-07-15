"""
トレーニング共通ユーティリティ
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TrainingMetrics:
    """トレーニングメトリクスのデータクラス"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    timestamp: float
    additional_metrics: Dict[str, float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp
        }
        if self.additional_metrics:
            result.update(self.additional_metrics)
        return result


class MetricsLogger:
    """メトリクス記録クラス"""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None
    ):
        """
        メトリクスロガーの初期化
        
        Args:
            log_dir: ログディレクトリ
            use_tensorboard: TensorBoardを使用するかどうか
            use_wandb: Weights & Biasesを使用するかどうか
            wandb_project: W&Bプロジェクト名
            wandb_config: W&B設定
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[TrainingMetrics] = []
        
        # TensorBoard設定
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # Weights & Biases設定
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project or "llm-training",
                config=wandb_config or {},
                dir=str(self.log_dir)
            )
    
    def log_metrics(self, metrics: TrainingMetrics):
        """メトリクスの記録"""
        self.metrics_history.append(metrics)
        
        # TensorBoardに記録
        if self.use_tensorboard:
            self.tb_writer.add_scalar("Loss/Train", metrics.loss, metrics.step)
            self.tb_writer.add_scalar("Learning_Rate", metrics.learning_rate, metrics.step)
            
            if metrics.additional_metrics:
                for key, value in metrics.additional_metrics.items():
                    self.tb_writer.add_scalar(f"Metrics/{key}", value, metrics.step)
        
        # W&Bに記録
        if self.use_wandb:
            wandb.log(metrics.to_dict(), step=metrics.step)
        
        # ファイルに保存
        self._save_metrics_to_file(metrics)
    
    def _save_metrics_to_file(self, metrics: TrainingMetrics):
        """メトリクスをファイルに保存"""
        metrics_file = self.log_dir / "training_metrics.jsonl"
        with open(metrics_file, 'a', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """トレーニング曲線をプロット"""
        if not self.metrics_history:
            self.logger.warning("プロットするメトリクスがありません")
            return
        
        steps = [m.step for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        learning_rates = [m.learning_rate for m in self.metrics_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss curve
        ax1.plot(steps, losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate curve
        ax2.plot(steps, learning_rates, 'r-', label='Learning Rate')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def get_summary_stats(self) -> Dict[str, float]:
        """サマリー統計を取得"""
        if not self.metrics_history:
            return {}
        
        losses = [m.loss for m in self.metrics_history]
        
        return {
            "total_steps": len(self.metrics_history),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "avg_loss": np.mean(losses),
            "loss_std": np.std(losses)
        }
    
    def close(self):
        """ロガーを閉じる"""
        if self.use_tensorboard and hasattr(self, 'tb_writer'):
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()


class EarlyStopping:
    """早期停止クラス"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ):
        """
        早期停止の初期化
        
        Args:
            patience: 改善が見られない最大エポック数
            min_delta: 改善と見なす最小変化量
            restore_best_weights: 最良の重みを復元するかどうか
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        早期停止の判定
        
        Args:
            val_loss: 検証損失
            model: モデル
            
        Returns:
            停止するかどうか
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class CheckpointManager:
    """チェックポイント管理クラス"""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        チェックポイントマネージャーの初期化
        
        Args:
            checkpoint_dir: チェックポイントディレクトリ
            max_checkpoints: 保持する最大チェックポイント数
            save_best_only: 最良のモデルのみ保存するかどうか
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        self.best_loss = float('inf')
        self.checkpoint_files: List[Path] = []
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        additional_state: Optional[Dict] = None
    ) -> Path:
        """
        チェックポイントを保存
        
        Args:
            model: モデル
            optimizer: オプティマイザー
            epoch: エポック
            loss: 損失
            additional_state: 追加の状態
            
        Returns:
            保存されたチェックポイントのパス
        """
        is_best = loss < self.best_loss
        
        if self.save_best_only and not is_best:
            return None
        
        if is_best:
            self.best_loss = loss
        
        # チェックポイントファイル名
        checkpoint_name = f"checkpoint_epoch_{epoch}_loss_{loss:.4f}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 状態を保存
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'is_best': is_best
        }
        
        if additional_state:
            state.update(additional_state)
        
        torch.save(state, checkpoint_path)
        
        # 最良モデルの別途保存
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
        
        # チェックポイントリストの管理
        self.checkpoint_files.append(checkpoint_path)
        
        # 古いチェックポイントの削除
        if not self.save_best_only and len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        self.logger.info(f"チェックポイントを保存しました: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        チェックポイントを読み込み
        
        Args:
            checkpoint_path: チェックポイントパス
            model: モデル
            optimizer: オプティマイザー
            
        Returns:
            読み込まれた状態
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint_path}")
        
        state = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(state['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        
        self.logger.info(f"チェックポイントを読み込みました: {checkpoint_path}")
        
        return state


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """ロギングの設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """モデルのパラメータ数をカウント"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def set_seed(seed: int = 42):
    """乱数シードの設定"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")