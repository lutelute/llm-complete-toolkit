#!/usr/bin/env python3
"""
学習済みモデルを自動でGGUF形式に変換するスクリプト
"""

import sys
import os
import subprocess
from pathlib import Path

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
            ], check=True, capture_output=True)
            print("✅ クローン完了")
        else:
            print("✅ llama.cppディレクトリが既に存在")
        
        # ビルド
        print("llama.cppをビルド中...")
        result = subprocess.run([
            "make", "-C", "./llama.cpp", "-j4"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ビルド完了")
        else:
            print(f"ビルド警告/エラー: {result.stderr}")
        
        # 必要なファイルの存在確認
        convert_script = "./llama.cpp/convert_hf_to_gguf.py"
        quantize_tool = "./llama.cpp/llama-quantize"
        
        if os.path.exists(convert_script):
            print(f"✅ 変換スクリプト確認: {convert_script}")
            return convert_script
        else:
            # 新しいパスも確認
            convert_script = "./llama.cpp/convert.py"
            if os.path.exists(convert_script):
                print(f"✅ 変換スクリプト確認: {convert_script}")
                return convert_script
            else:
                print("❌ 変換スクリプトが見つかりません")
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
        
        # convert_hf_to_gguf.py または convert.py を使用
        if "convert_hf_to_gguf.py" in convert_script:
            cmd = [
                "python3", convert_script,
                model_path,
                "--outdir", os.path.dirname(output_path),
                "--outfile", os.path.basename(output_path)
            ]
        else:
            # 新しいconvert.pyの場合
            cmd = [
                "python3", convert_script,
                model_path,
                "--outdir", os.path.dirname(output_path)
            ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ GGUF変換完了")
            print(f"stdout: {result.stdout}")
            
            # 出力ファイルを探す
            output_dir = os.path.dirname(output_path)
            for file in os.listdir(output_dir):
                if file.endswith('.gguf'):
                    actual_output = os.path.join(output_dir, file)
                    size_mb = os.path.getsize(actual_output) / (1024 * 1024)
                    print(f"出力ファイル: {actual_output} ({size_mb:.1f} MB)")
                    return actual_output
            
            return output_path if os.path.exists(output_path) else None
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
            return gguf_path  # 元のファイルを返す
        
        cmd = [
            quantize_tool,
            gguf_path,
            quantized_path,
            quantization_type
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 量子化完了")
            
            if os.path.exists(quantized_path):
                size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
                print(f"量子化ファイル: {quantized_path} ({size_mb:.1f} MB)")
                return quantized_path
            else:
                print("❌ 量子化ファイルが見つかりません")
                return gguf_path
        else:
            print(f"量子化エラー（元ファイルを使用）:")
            print(f"stderr: {result.stderr}")
            return gguf_path
            
    except subprocess.TimeoutExpired:
        print("❌ 量子化がタイムアウトしました（元ファイルを使用）")
        return gguf_path
    except Exception as e:
        print(f"量子化エラー: {e}（元ファイルを使用）")
        return gguf_path

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
        
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"✅ コピー完了: {target_path} ({size_mb:.1f} MB)")
        return target_path
        
    except Exception as e:
        print(f"❌ コピーエラー: {e}")
        return None

def main():
    print("=== 自動GGUF変換ツール ===")
    
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
        print("先にmerge_for_lmstudio.pyまたはmerge_gfm_model.pyを実行してください")
        return
    
    print(f"\n利用可能なモデル:")
    for i, (name, path) in enumerate(available_models):
        print(f"  {i+1}. {name} ({path})")
    
    # llama.cppのインストール
    convert_script = install_llama_cpp()
    if not convert_script:
        print("❌ llama.cppのインストールに失敗しました")
        return
    
    # 全モデルを変換
    successful_conversions = []
    
    for name, model_path in available_models:
        print(f"\n{'='*60}")
        print(f"=== {name}モデルの変換開始 ===")
        print(f"{'='*60}")
        
        # GGUF変換
        gguf_dir = "./gguf_models"
        gguf_path = os.path.join(gguf_dir, f"{name}.gguf")
        
        converted_path = convert_to_gguf(model_path, gguf_path, convert_script)
        
        if converted_path and os.path.exists(converted_path):
            # 量子化
            quantized_path = quantize_gguf(converted_path, "Q4_K_M")
            
            # LM Studioにコピー
            lm_studio_path = copy_to_lm_studio(quantized_path, f"{name}-gguf")
            
            if lm_studio_path:
                successful_conversions.append((name, lm_studio_path))
                print(f"✅ {name}モデルのGGUF変換完了")
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
        print(f"2. 以下のGGUFモデルが利用可能:")
        for name, _ in successful_conversions:
            print(f"   - {name}-gguf")
        
        print(f"\n💡 GGUF形式の利点:")
        print(f"- より高速な推論速度")
        print(f"- メモリ使用量の削減")
        print(f"- CPU推論の最適化")
        print(f"- 量子化による小さなファイルサイズ")
        
    else:
        print(f"❌ 変換に成功したモデルはありません")
    
    print(f"\n🎉 GGUF変換処理完了！")

if __name__ == "__main__":
    main()