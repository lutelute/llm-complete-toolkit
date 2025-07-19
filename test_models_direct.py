#!/usr/bin/env python3
"""
学習済みモデルを直接テストするスクリプト
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_model(model_path: str, model_name: str):
    """モデルのテスト"""
    try:
        print(f"=== {model_name}モデルのテスト開始 ===")
        print(f"モデルパス: {model_path}")
        
        # デバイス設定
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"使用デバイス: {device}")
        
        # モデルとトークナイザーの読み込み
        print("モデルとトークナイザーを読み込み中...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map="auto" if device != "cpu" else None
        )
        
        if device == "mps":
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ モデル読み込み完了 ({load_time:.2f}秒)")
        
        # パラメータ数を計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"総パラメータ数: {total_params:,}")
        print(f"訓練可能パラメータ数: {trainable_params:,}")
        
        # テストプロンプト設定
        if "GFM" in model_name:
            test_prompts = [
                "グリッドフォーミングインバータとは何ですか？",
                "再生可能エネルギーの課題について説明してください。",
                "マイクログリッドの利点は何ですか？"
            ]
        else:
            test_prompts = [
                "Pythonでprint文を使って挨拶を出力する方法を教えて",
                "機械学習とは何ですか？",
                "プログラミングの基本について説明してください"
            ]
        
        # 各プロンプトでテスト
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- テスト {i}: {prompt[:30]}... ---")
            
            try:
                # トークン化
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                if device == "mps":
                    inputs = inputs.to(device)
                
                # 生成開始
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generation_time = time.time() - start_time
                
                # デコード
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                
                print(f"入力: {prompt}")
                print(f"出力: {response}")
                print(f"生成時間: {generation_time:.2f}秒")
                
                # トークン/秒の計算
                new_tokens = len(outputs[0]) - len(inputs[0])
                tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
                print(f"生成速度: {tokens_per_sec:.1f} tokens/sec")
                
            except Exception as e:
                print(f"❌ テスト{i}でエラー: {e}")
        
        print(f"✅ {model_name}モデルのテスト完了")
        return True
        
    except Exception as e:
        print(f"❌ {model_name}モデルのテストでエラー: {e}")
        return False

def benchmark_models():
    """全モデルのベンチマーク"""
    print("=== 学習済みモデルベンチマーク ===")
    
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
        print("❌ テスト可能なモデルが見つかりません")
        return
    
    print(f"\n利用可能なモデル:")
    for i, (name, path) in enumerate(available_models):
        print(f"  {i+1}. {name} ({path})")
    
    # システム情報
    print(f"\n=== システム情報 ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS利用可能: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPSビルト: {torch.backends.mps.is_built()}")
    
    # 全モデルをテスト
    successful_tests = []
    
    for name, model_path in available_models:
        print(f"\n{'='*60}")
        success = test_model(model_path, name)
        if success:
            successful_tests.append(name)
        print(f"{'='*60}")
    
    # 結果サマリー
    print(f"\n=== ベンチマーク結果サマリー ===")
    
    if successful_tests:
        print(f"✅ 成功したテスト: {len(successful_tests)}個")
        for name in successful_tests:
            print(f"  - {name}")
        
        print(f"\n🚀 推奨使用方法:")
        print(f"1. 直接Pythonスクリプトから利用")
        print(f"2. 上記のテスト結果を参考に適切なプロンプトを使用")
        print(f"3. Apple Siliconの場合はMPSデバイスで高速動作")
        
        print(f"\n💡 最適化のポイント:")
        print(f"- 🍎 MPSデバイス利用でApple Silicon最適化")
        print(f"- 🧠 float16精度でメモリ効率化")
        print(f"- ⚡ バッチ処理で高速化可能")
        print(f"- 🎯 ドメイン特化モデルで高品質回答")
        
    else:
        print(f"❌ 成功したテストはありません")
    
    print(f"\n🎉 ベンチマーク完了！")

def main():
    benchmark_models()

if __name__ == "__main__":
    main()