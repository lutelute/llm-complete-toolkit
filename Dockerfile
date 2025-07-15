# LLM Complete Toolkit Docker Image
FROM python:3.10-slim

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# 環境変数の設定
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルのコピー
COPY . .

# セットアップスクリプトの実行
RUN python setup.py

# デフォルトコマンド
CMD ["python", "main.py", "--help"]