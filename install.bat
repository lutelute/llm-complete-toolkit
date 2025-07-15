@echo off
REM LLM Complete Toolkit インストールスクリプト (Windows)

echo 🚀 LLM Complete Toolkit インストール開始

REM Python バージョンチェック
echo [INFO] Python バージョンを確認中...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python が見つかりません
    echo Python 3.8以上をインストールしてください
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python バージョン: %PYTHON_VERSION%

REM Python 3.8以上のチェック
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8以上が必要です
    pause
    exit /b 1
)

echo [SUCCESS] Python バージョン OK

REM 仮想環境の作成
echo [INFO] 仮想環境をセットアップ中...
if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] 仮想環境作成完了
) else (
    echo [INFO] 仮想環境は既に存在します
)

REM 仮想環境の有効化
call venv\Scripts\activate
echo [SUCCESS] 仮想環境を有効化

REM pipのアップグレード
echo [INFO] pipをアップグレード中...
python -m pip install --upgrade pip setuptools wheel
echo [SUCCESS] pipアップグレード完了

REM 依存関係のインストール
echo [INFO] 必要なパッケージをインストール中...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] パッケージインストールに失敗
    pause
    exit /b 1
)
echo [SUCCESS] パッケージインストール完了

REM GPU サポートの確認
echo [INFO] GPU サポートを確認中...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if %errorlevel% equ 0 (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] CUDA GPU サポート利用可能
        
        REM Flash Attention のインストール（オプション）
        echo [INFO] Flash Attention をインストール中...
        pip install flash-attn --no-build-isolation
        if %errorlevel% equ 0 (
            echo [SUCCESS] Flash Attention インストール完了
        ) else (
            echo [WARNING] Flash Attention のインストールに失敗（オプション）
        )
    ) else (
        echo [INFO] GPU利用不可、CPUモードで実行します
    )
) else (
    echo [WARNING] PyTorch がインストールされていません
)

REM ディレクトリの作成
echo [INFO] 必要なディレクトリを作成中...
mkdir data 2>nul
mkdir outputs 2>nul
mkdir models 2>nul
mkdir models\base_models 2>nul
mkdir models\fine_tuned_models 2>nul
mkdir models\trained_models 2>nul
mkdir logs 2>nul
mkdir examples 2>nul
echo [SUCCESS] ディレクトリ作成完了

REM 基本テスト
echo [INFO] 基本テストを実行中...
python setup.py
if %errorlevel% neq 0 (
    echo [ERROR] セットアップテストに失敗
    pause
    exit /b 1
)
echo [SUCCESS] セットアップテスト完了

echo ================================================
echo [SUCCESS] インストール完了！
echo ================================================
echo.
echo 次のステップ:
echo 1. venv\Scripts\activate          # 仮想環境を有効化
echo 2. python main.py --help         # コマンドを確認
echo 3. python scripts\download_models.py --list-popular  # モデル一覧を確認
echo.
echo 使用例:
echo python main.py create-samples                    # サンプルデータ作成
echo python main.py extract data\ outputs\           # ドキュメント処理
echo python main.py train-lora --train-data data\train.jsonl  # LoRA学習
echo.
pause