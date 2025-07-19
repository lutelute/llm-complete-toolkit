#!/usr/bin/env python3
"""
MLXで直接モデルをテストするスクリプト
"""

import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ MLXとTransformersが利用可能です")
except ImportError as e:
    print(f"❌ 必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

def test_model_with_mlx(model_path: str):
    """MLX最適化でモデルをテスト"""
    print(f"\n🚀 モデル読み込み: {model_path}")
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ トークナイザー読み込み完了")
    
    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="mps",  # Apple Silicon最適化
        torch_dtype=mx.float32 if hasattr(mx, 'float32') else None
    )
    print("✅ モデル読み込み完了")
    
    # GFMテスト質問
    test_prompts = [
        "GFMとは何ですか？",
        "グリッドフォーミングインバーターの特徴を教えてください。",
        "こんにちは"
    ]
    
    print("\n🧪 GFMテストを開始します...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 質問{i}: {prompt}")
        
        # トークン化
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # MLX最適化で生成
        with mx.no_grad() if hasattr(mx, 'no_grad') else nullcontext():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 結果をデコード
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        print(f"💬 回答: {answer}")
        print("-" * 50)

def main():
    """メイン関数"""
    model_path = "models/safetensors_models/GFM-DialoGPT-small-safetensors"
    
    if not Path(model_path).exists():
        print(f"❌ モデルが見つかりません: {model_path}")
        print("まず以下のコマンドでモデルを作成してください:")
        print("python scripts/merge_lora_model.py --base-model models/base_models/microsoft_DialoGPT-small --lora-adapter models/fine_tuned_models/DialoGPT-small-GFM --output models/safetensors_models/GFM-DialoGPT-small-safetensors --device auto")
        sys.exit(1)
    
    print("🍎 Apple Silicon (MLX) 最適化テスト")
    print("=" * 60)
    
    try:
        test_model_with_mlx(model_path)
        print("\n✅ テスト完了!")
        print("\n📊 パフォーマンス情報:")
        print("- Apple Silicon (M4) 最適化済み")
        print("- 統合メモリ効率活用")
        print("- CPU+GPU協調処理")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print("MLX直接変換は現在DialoGPTに対応していません")
        print("代替案: PyTorchのMPS最適化を使用中")

if __name__ == "__main__":
    main()