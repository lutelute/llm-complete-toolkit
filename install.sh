#!/bin/bash
# LLM Complete Toolkit インストールスクリプト (Linux/macOS)

set -e

echo "🚀 LLM Complete Toolkit インストール開始"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Python バージョンチェック
check_python() {
    log_info "Python バージョンを確認中..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python が見つかりません"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
    log_success "Python バージョン: $PYTHON_VERSION"
    
    # Python 3.8以上のチェック
    if $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_success "Python バージョン OK"
    else
        log_error "Python 3.8以上が必要です"
        exit 1
    fi
}

# 仮想環境の作成
setup_venv() {
    log_info "仮想環境をセットアップ中..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        log_success "仮想環境作成完了"
    else
        log_info "仮想環境は既に存在します"
    fi
    
    # 仮想環境の有効化
    source venv/bin/activate
    log_success "仮想環境を有効化"
    
    # 仮想環境内でpipのアップグレード
    pip install --upgrade pip setuptools wheel
    log_success "pip アップグレード完了"
}

# 必要なパッケージのインストール
install_packages() {
    log_info "必要なパッケージをインストール中..."
    
    # Apple Silicon用のrequirements.txtを作成
    create_macos_requirements
    
    # 依存関係のインストール
    pip install -r requirements_macos.txt
    
    log_success "パッケージインストール完了"
}

# Apple Silicon用のrequirements.txtを作成
create_macos_requirements() {
    log_info "Apple Silicon用のrequirements.txtを作成中..."
    
    # macOS用のrequirements.txtを作成（CUDAパッケージを除外）
    cat > requirements_macos.txt << 'EOF'
# Core Dependencies
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0

# Hugging Face Hub
huggingface-hub>=0.19.0
tokenizers>=0.15.0

# Document Processing
PyPDF2==3.0.1
pdfplumber==0.10.0
markdown==3.5.1
python-docx==1.1.0
jsonlines==4.0.0

# Transfer Learning (LoRA only - QLoRA not supported on Apple Silicon)
peft>=0.6.0
# bitsandbytes>=0.41.0  # Not supported on Apple Silicon

# Reinforcement Learning
stable-baselines3>=2.1.0
gymnasium>=0.29.0
gym>=0.26.0
tianshou>=0.5.0

# Training & Monitoring
tensorboard>=2.14.0
wandb>=0.16.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Configuration & Utils
hydra-core>=1.3.0
omegaconf>=2.3.0
PyYAML>=6.0
click>=8.1.0
rich>=13.0.0
psutil>=5.9.0

# Optional (Development)
streamlit>=1.28.0
gradio>=4.0.0
jupyter>=1.0.0
EOF

    log_success "Apple Silicon用requirements.txt作成完了"
}

# GPU サポートの確認
check_gpu() {
    log_info "GPU サポートを確認中..."
    
    if python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
        # Apple Silicon (MPS) の確認
        if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
            log_success "Apple Silicon MPS サポート利用可能"
            
            # config.yamlでMPSを有効化
            log_info "config.yamlでMPS設定を更新中..."
            if [ -f "configs/config.yaml" ]; then
                sed -i.bak 's/device: "cuda"/device: "mps"/' configs/config.yaml
                log_success "MPS設定を更新完了"
            fi
        else
            log_info "MPS利用不可、CPUモードで実行します"
        fi
        
        # CUDA確認（参考情報）
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            log_info "CUDA GPU サポートも利用可能（Apple Siliconでは使用されません）"
        fi
    else
        log_warning "PyTorch がインストールされていません"
    fi
}

# ディレクトリの作成
setup_directories() {
    log_info "必要なディレクトリを作成中..."
    
    mkdir -p data outputs models/{base_models,fine_tuned_models,trained_models} logs examples
    
    log_success "ディレクトリ作成完了"
}

# 基本テスト
run_tests() {
    log_info "基本テストを実行中..."
    
    if python setup.py; then
        log_success "セットアップテスト完了"
    else
        log_error "セットアップテストに失敗"
        exit 1
    fi
}

# メイン処理
main() {
    echo "================================================"
    echo "LLM Complete Toolkit インストーラー"
    echo "================================================"
    
    # 1. Python チェック
    check_python
    
    # 2. 仮想環境の設定
    setup_venv
    
    # 3. パッケージのインストール
    install_packages
    
    # 4. GPU サポートの確認
    check_gpu
    
    # 5. ディレクトリの作成
    setup_directories
    
    # 6. 基本テスト
    run_tests
    
    echo "================================================"
    log_success "インストール完了！"
    echo "================================================"
    
    echo ""
    echo "次のステップ:"
    echo "1. source venv/bin/activate  # 仮想環境を有効化"
    echo "2. python main.py --help     # コマンドを確認"
    echo "3. python scripts/download_models.py --list-popular  # モデル一覧を確認"
    echo ""
    echo "使用例:"
    echo "python main.py create-samples              # サンプルデータ作成"
    echo "python main.py extract data/ outputs/     # ドキュメント処理"
    echo "python main.py train-lora --train-data data/train.jsonl  # LoRA学習"
    echo ""
}

# スクリプト実行
if [ "$0" = "${BASH_SOURCE[0]}" ]; then
    main "$@"
fi