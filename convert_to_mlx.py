#!/usr/bin/env python3
"""
学習済みモデルをMLX形式に変換するスクリプト
Apple Silicon最適化用
"""

import sys
import os
import subprocess
from pathlib import Path

def install_mlx():
    """MLXライブラリのインストール"""
    try:
        print("=== MLXライブラリのインストール ===")
        
        # MLX関連パッケージをインストール
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "mlx", "mlx-lm", "--upgrade"
        ], check=True)
        
        print("✅ MLXライブラリインストール完了")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ MLXインストールエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False

def convert_to_mlx(model_path: str, output_path: str):
    """モデルをMLX形式に変換"""
    try:
        print(f"=== MLX変換開始 ===")
        print(f"入力: {model_path}")
        print(f"出力: {output_path}")
        
        # 既存の出力ディレクトリを削除
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
            print(f"既存ディレクトリを削除: {output_path}")
        
        # 出力ディレクトリを作成
        os.makedirs(output_path, exist_ok=True)
        
        # MLX変換コマンド（新しい形式）
        cmd = [
            "python", "-m", "mlx_lm", "convert",
            "--hf-path", model_path,
            "--mlx-path", output_path,
            "--quantize"
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ MLX変換完了")
            print(f"stdout: {result.stdout[-500:]}")  # 最後の500文字のみ表示
            
            # 出力ファイルサイズを確認
            total_size = 0
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            size_mb = total_size / (1024 * 1024)
            print(f"出力ディレクトリ: {output_path} (合計 {size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ 変換エラー:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 変換がタイムアウトしました")
        return False
    except Exception as e:
        print(f"❌ 変換中にエラーが発生: {e}")
        return False

def copy_to_lm_studio(mlx_path: str, model_name: str):
    """MLXファイルをLM Studioディレクトリにコピー"""
    try:
        import shutil
        
        lm_studio_dir = os.path.expanduser("~/Library/Application Support/lm-studio/models")
        target_dir = os.path.join(lm_studio_dir, f"{model_name}-mlx")
        
        print(f"=== LM Studioにコピー ===")
        print(f"コピー先: {target_dir}")
        
        # 既存ディレクトリを削除して新しく作成
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # ディレクトリ全体をコピー
        shutil.copytree(mlx_path, target_dir)
        
        # サイズを計算
        total_size = 0
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        size_mb = total_size / (1024 * 1024)
        print(f"✅ コピー完了: {target_dir} (合計 {size_mb:.1f} MB)")
        return target_dir
        
    except Exception as e:
        print(f"❌ コピーエラー: {e}")
        return None

def test_mlx_model(mlx_path: str, model_name: str):
    """MLXモデルのテスト"""
    try:
        print(f"=== {model_name}モデルのテスト ===")
        
        # テストプロンプト
        if "GFM" in model_name:
            test_prompt = "グリッドフォーミングインバータとは何ですか？"
        else:
            test_prompt = "Pythonでprint文を使って挨拶を出力する方法を教えて"
        
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", mlx_path,
            "--prompt", test_prompt,
            "--max-tokens", "50"
        ]
        
        print(f"テストコマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ MLXモデルテスト成功")
            print(f"出力: {result.stdout}")
            return True
        else:
            print(f"❌ テストエラー: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ テストがタイムアウトしました")
        return False
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        return False

def main():
    print("=== MLX変換ツール（Apple Silicon最適化） ===")
    
    # 利用可能なモデルを確認
    model_paths = {
        "minimal_test": "./lm_studio_models/minimal_test_merged",
        "GFM": "./lm_studio_models/DialoGPT-small-GFM_merged"
    }
    
    available_models = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            available_models.append((name, path))
        else:
            print(f"スキップ: {name} (パスが存在しません: {path})")
    
    if not available_models:
        print("❌ 変換可能なモデルが見つかりません")
        return
    
    print(f"\n利用可能なモデル:")
    for i, (name, path) in enumerate(available_models):
        print(f"  {i+1}. {name} ({path})")
    
    # MLXのインストール確認
    try:
        import mlx
        import mlx_lm
        print("✅ MLXライブラリが利用可能")
    except ImportError:
        print("MLXライブラリをインストール中...")
        if not install_mlx():
            print("❌ MLXのインストールに失敗しました")
            return
    
    # 全モデルを変換
    successful_conversions = []
    
    for name, model_path in available_models:
        print(f"\n{'='*60}")
        print(f"=== {name}モデルの変換開始 ===")
        print(f"{'='*60}")
        
        # MLX変換
        mlx_dir = "./mlx_models"
        mlx_path = os.path.join(mlx_dir, name)
        
        success = convert_to_mlx(model_path, mlx_path)
        
        if success and os.path.exists(mlx_path):
            # LM Studioにコピー
            lm_studio_path = copy_to_lm_studio(mlx_path, name)
            
            if lm_studio_path:
                # モデルテスト
                test_success = test_mlx_model(mlx_path, name)
                
                successful_conversions.append((name, lm_studio_path))
                print(f"✅ {name}モデルのMLX変換完了")
            else:
                print(f"❌ {name}モデルのLM Studioコピー失敗")
        else:
            print(f"❌ {name}モデルの変換失敗")
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"=== 変換結果サマリー ===")
    print(f"{'='*60}")
    
    if successful_conversions:
        print(f"✅ 成功した変換: {len(successful_conversions)}個")
        for name, path in successful_conversions:
            print(f"  - {name}: {path}")
        
        print(f"\n🚀 LM Studio使用方法:")
        print(f"1. LM Studioを開く")
        print(f"2. 以下のMLXモデルが利用可能:")
        for name, _ in successful_conversions:
            print(f"   📁 {name}-mlx")
        
        print(f"\n💡 MLX形式の利点:")
        print(f"- 🍎 Apple Silicon専用最適化")
        print(f"- ⚡ 超高速推論速度")
        print(f"- 🧠 統合メモリ活用")
        print(f"- 🔋 省電力動作")
        print(f"- 🎯 Metal Performance Shaders利用")
        
        print(f"\n🧪 テスト用プロンプト:")
        print(f"GFMモデル: 'グリッドフォーミングインバータとは何ですか？'")
        print(f"minimal_testモデル: 'Pythonでprint文を使って挨拶を出力する方法を教えて'")
        
        print(f"\n📋 MLXモデル直接利用:")
        for name, _ in successful_conversions:
            print(f"python -m mlx_lm.generate --model ./mlx_models/{name} --prompt 'あなたのプロンプト'")
        
    else:
        print(f"❌ 変換に成功したモデルはありません")
    
    print(f"\n🎉 MLX変換処理完了！")

if __name__ == "__main__":
    main()