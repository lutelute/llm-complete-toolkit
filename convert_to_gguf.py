#!/usr/bin/env python3
"""
学習済みモデルをGGUF形式に変換するスクリプト
"""

import sys
import os
import subprocess
from pathlib import Path
import json

def check_llama_cpp():
    """llama.cppの存在確認とインストール"""
    try:
        # llama.cppのconvert.pyを探す
        llama_cpp_paths = [
            "/usr/local/bin/llama.cpp",
            "/opt/homebrew/bin/llama.cpp", 
            "~/llama.cpp",
            "./llama.cpp"
        ]
        
        for path in llama_cpp_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                convert_script = os.path.join(expanded_path, "convert_hf_to_gguf.py")
                if os.path.exists(convert_script):
                    print(f"✅ llama.cpp found: {expanded_path}")
                    return convert_script
        
        print("❌ llama.cppが見つかりません")
        print("以下の手順でインストールしてください：")
        print("1. git clone https://github.com/ggerganov/llama.cpp")
        print("2. cd llama.cpp")
        print("3. make")
        return None
        
    except Exception as e:
        print(f"llama.cpp確認エラー: {e}")
        return None

def install_llama_cpp():
    """llama.cppを自動インストール"""
    try:
        print("=== llama.cpp自動インストール開始 ===")
        
        # 現在のディレクトリにクローン
        if not os.path.exists("./llama.cpp"):
            print("llama.cppをクローン中...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/ggerganov/llama.cpp.git"
            ], check=True)
        
        # ビルド
        print("llama.cppをビルド中...")
        subprocess.run([
            "make", "-C", "./llama.cpp"
        ], check=True)
        
        convert_script = "./llama.cpp/convert_hf_to_gguf.py"
        if os.path.exists(convert_script):
            print("✅ llama.cppインストール完了")
            return convert_script
        else:
            print("❌ convert_hf_to_gguf.pyが見つかりません")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ llama.cppインストールエラー: {e}")
        return None
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return None

def convert_to_gguf(model_path: str, output_path: str, convert_script: str):
    """モデルをGGUF形式に変換"""
    try:
        print(f"=== GGUF変換開始 ===")
        print(f"入力: {model_path}")
        print(f"出力: {output_path}")
        
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 変換実行
        cmd = [
            "python3", convert_script,
            model_path,
            "--outdir", os.path.dirname(output_path),
            "--outfile", os.path.basename(output_path)
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ GGUF変換完了")
            
            # ファイルサイズを確認
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"出力ファイル: {output_path} ({size_mb:.1f} MB)")
                return True
            else:
                print("❌ 出力ファイルが見つかりません")
                return False
        else:
            print(f"❌ 変換エラー:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 変換中にエラーが発生: {e}")
        return False

def quantize_gguf(gguf_path: str, quantization_type: str = "Q4_K_M"):
    """GGUFモデルを量子化"""
    try:
        print(f"=== GGUF量子化開始 ({quantization_type}) ===")
        
        # 量子化後のファイル名
        base_name = os.path.splitext(gguf_path)[0]
        quantized_path = f"{base_name}-{quantization_type}.gguf"
        
        # llama.cppの量子化ツール
        quantize_tool = "./llama.cpp/llama-quantize"
        
        if not os.path.exists(quantize_tool):
            print("❌ llama-quantizeが見つかりません")
            return None
        
        cmd = [
            quantize_tool,
            gguf_path,
            quantized_path,
            quantization_type
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 量子化完了")
            
            if os.path.exists(quantized_path):
                size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
                print(f"量子化ファイル: {quantized_path} ({size_mb:.1f} MB)")
                return quantized_path
            else:
                print("❌ 量子化ファイルが見つかりません")
                return None
        else:
            print(f"❌ 量子化エラー:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ 量子化中にエラーが発生: {e}")
        return None

def copy_to_lm_studio(gguf_path: str, model_name: str):
    """GGUFファイルをLM Studioディレクトリにコピー"""
    try:
        import shutil
        
        lm_studio_dir = os.path.expanduser("~/Library/Application Support/lm-studio/models")
        target_dir = os.path.join(lm_studio_dir, model_name)
        
        print(f"=== LM Studioにコピー ===")
        print(f"コピー先: {target_dir}")
        
        # ディレクトリを作成
        os.makedirs(target_dir, exist_ok=True)
        
        # ファイルをコピー
        target_path = os.path.join(target_dir, os.path.basename(gguf_path))
        shutil.copy2(gguf_path, target_path)
        
        print(f"✅ コピー完了: {target_path}")
        return target_path
        
    except Exception as e:
        print(f"❌ コピーエラー: {e}")
        return None

def main():
    print("=== モデルGGUF変換ツール ===")
    
    # 利用可能なモデルを確認
    model_paths = {
        "minimal_test": "./lm_studio_models/minimal_test_merged",
        "GFM": "./lm_studio_models/DialoGPT-small-GFM_merged"
    }
    
    available_models = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            available_models.append((name, path))
    
    if not available_models:
        print("❌ 変換可能なモデルが見つかりません")
        print("先にmerge_for_lmstudio.pyまたはmerge_gfm_model.pyを実行してください")
        return
    
    print(f"\n利用可能なモデル:")
    for i, (name, path) in enumerate(available_models):
        print(f"  {i+1}. {name} ({path})")
    
    # llama.cppの確認
    convert_script = check_llama_cpp()
    if not convert_script:
        print("\nllama.cppを自動インストールしますか？ (y/n)")
        if input().lower() == 'y':
            convert_script = install_llama_cpp()
            if not convert_script:
                print("❌ llama.cppのインストールに失敗しました")
                return
        else:
            print("❌ llama.cppが必要です")
            return
    
    # 全モデルを変換
    for name, model_path in available_models:
        print(f"\n=== {name}モデルの変換開始 ===")
        
        # GGUF変換
        gguf_dir = "./gguf_models"
        gguf_path = os.path.join(gguf_dir, f"{name}.gguf")
        
        success = convert_to_gguf(model_path, gguf_path, convert_script)
        
        if success:
            # 量子化
            quantized_path = quantize_gguf(gguf_path, "Q4_K_M")
            
            if quantized_path:
                # LM Studioにコピー
                lm_studio_path = copy_to_lm_studio(quantized_path, f"{name}-gguf")
                
                if lm_studio_path:
                    print(f"✅ {name}モデルのGGUF変換完了")
                    print(f"LM Studioパス: {lm_studio_path}")
        
        print("-" * 50)
    
    print(f"\n🎉 GGUF変換完了！")
    print(f"\n🚀 LM Studio使用方法:")
    print(f"1. LM Studioを開く")
    print(f"2. 以下のモデルが利用可能:")
    for name, _ in available_models:
        print(f"   - {name}-gguf")
    print(f"\n💡 GGUFの利点:")
    print(f"- より高速な推論")
    print(f"- メモリ使用量削減")
    print(f"- CPU推論の最適化")

if __name__ == "__main__":
    main()