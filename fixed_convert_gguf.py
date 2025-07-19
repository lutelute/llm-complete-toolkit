#!/usr/bin/env python3
"""
学習済みモデルをGGUF形式に変換するスクリプト（修正版）
"""

import sys
import os
import subprocess
from pathlib import Path

def convert_to_gguf_fixed(model_path: str, output_path: str, convert_script: str):
    """モデルをGGUF形式に変換（修正版）"""
    try:
        print(f"=== GGUF変換開始 ===")
        print(f"入力: {model_path}")
        print(f"出力: {output_path}")
        
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 正しいパラメータで実行
        cmd = [
            "python3", convert_script,
            model_path,
            "--outfile", output_path
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ GGUF変換完了")
            print(f"stdout: {result.stdout[-500:]}")  # 最後の500文字のみ表示
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"出力ファイル: {output_path} ({size_mb:.1f} MB)")
                return output_path
            else:
                print("❌ 出力ファイルが見つかりません")
                return None
        else:
            print(f"❌ 変換エラー:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ 変換がタイムアウトしました")
        return None
    except Exception as e:
        print(f"❌ 変換中にエラーが発生: {e}")
        return None

def build_llama_cpp_cmake():
    """CMakeを使ってllama.cppをビルド"""
    try:
        print("=== CMakeでllama.cppをビルド ===")
        
        # build ディレクトリを作成
        build_dir = "./llama.cpp/build"
        os.makedirs(build_dir, exist_ok=True)
        
        # CMake設定
        print("CMake設定中...")
        cmake_config = subprocess.run([
            "cmake", "..", "-DCMAKE_BUILD_TYPE=Release"
        ], cwd=build_dir, capture_output=True, text=True)
        
        if cmake_config.returncode != 0:
            print(f"CMake設定エラー: {cmake_config.stderr}")
            return False
        
        # ビルド実行
        print("ビルド実行中...")
        build_result = subprocess.run([
            "cmake", "--build", ".", "--config", "Release", "-j", "4"
        ], cwd=build_dir, capture_output=True, text=True)
        
        if build_result.returncode == 0:
            print("✅ CMakeビルド完了")
            return True
        else:
            print(f"ビルドエラー: {build_result.stderr}")
            return False
            
    except Exception as e:
        print(f"CMakeビルドエラー: {e}")
        return False

def main():
    print("=== 修正版GGUF変換ツール ===")
    
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
    
    # llama.cppの確認
    convert_script = "./llama.cpp/convert_hf_to_gguf.py"
    
    if not os.path.exists(convert_script):
        print("❌ llama.cppが見つかりません")
        return
    
    # CMakeでビルドを試行
    if not os.path.exists("./llama.cpp/build"):
        print("CMakeビルドを実行中...")
        build_llama_cpp_cmake()
    
    # 全モデルを変換
    successful_conversions = []
    
    for name, model_path in available_models:
        print(f"\n{'='*60}")
        print(f"=== {name}モデルの変換開始 ===")
        print(f"{'='*60}")
        
        # GGUF変換
        gguf_dir = "./gguf_models"
        gguf_path = os.path.join(gguf_dir, f"{name}.gguf")
        
        converted_path = convert_to_gguf_fixed(model_path, gguf_path, convert_script)
        
        if converted_path and os.path.exists(converted_path):
            # LM Studioにコピー
            try:
                import shutil
                
                lm_studio_dir = os.path.expanduser("~/Library/Application Support/lm-studio/models")
                target_dir = os.path.join(lm_studio_dir, f"{name}-gguf")
                target_path = os.path.join(target_dir, os.path.basename(converted_path))
                
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(converted_path, target_path)
                
                size_mb = os.path.getsize(target_path) / (1024 * 1024)
                print(f"✅ LM Studioコピー完了: {target_path} ({size_mb:.1f} MB)")
                
                successful_conversions.append((name, target_path))
                
            except Exception as e:
                print(f"❌ LM Studioコピーエラー: {e}")
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
        print(f"2. 以下のGGUFモデルが利用可能:")
        for name, _ in successful_conversions:
            print(f"   📁 {name}-gguf")
        
        print(f"\n💡 GGUF形式の利点:")
        print(f"- 🚀 より高速な推論速度")
        print(f"- 💾 メモリ使用量の削減")
        print(f"- 🖥️ CPU推論の最適化")
        print(f"- 📦 効率的なファイル形式")
        
        print(f"\n🧪 テスト用プロンプト:")
        print(f"GFMモデル: 'グリッドフォーミングインバータとは何ですか？'")
        print(f"minimal_testモデル: 'Pythonでprint文を使って挨拶を出力する方法を教えて'")
        
    else:
        print(f"❌ 変換に成功したモデルはありません")
    
    print(f"\n🎉 GGUF変換処理完了！")

if __name__ == "__main__":
    main()