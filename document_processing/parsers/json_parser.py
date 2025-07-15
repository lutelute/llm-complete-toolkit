#!/usr/bin/env python3
"""
JSON/JSONL パーサー
JSONファイルやJSONLファイルからデータを抽出し、学習データ形式に変換する
"""

import json
import jsonlines
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class Document:
    """ドキュメントを表すデータクラス"""
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str = "json"


class JsonParser:
    """JSON/JSONL ファイルパーサー"""
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Args:
            encoding: ファイルエンコーディング
        """
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)
        
    def parse(self, file_path: Union[str, Path]) -> List[Document]:
        """
        JSON/JSONLファイルを解析
        
        Args:
            file_path: ファイルパス
            
        Returns:
            List[Document]: 解析されたドキュメントリスト
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
        self.logger.info(f"JSONファイル解析開始: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.jsonl':
                return self._parse_jsonl(file_path)
            elif file_path.suffix.lower() == '.json':
                return self._parse_json(file_path)
            else:
                raise ValueError(f"サポートされていないファイル形式: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"JSON解析エラー {file_path}: {e}")
            raise
            
    def _parse_json(self, file_path: Path) -> List[Document]:
        """
        JSONファイルを解析
        
        Args:
            file_path: JSONファイルパス
            
        Returns:
            List[Document]: 解析されたドキュメントリスト
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
                
            # データの形式を判定して処理
            if isinstance(data, list):
                # 配列形式の場合
                for i, item in enumerate(data):
                    doc = self._process_json_item(item, file_path, i)
                    if doc:
                        documents.append(doc)
                        
            elif isinstance(data, dict):
                # 単一オブジェクト形式の場合
                doc = self._process_json_item(data, file_path, 0)
                if doc:
                    documents.append(doc)
                    
            else:
                self.logger.warning(f"予期しないJSON形式: {type(data)}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析エラー {file_path}: {e}")
            raise
            
        self.logger.info(f"JSON解析完了: {len(documents)}件のドキュメント")
        return documents
        
    def _parse_jsonl(self, file_path: Path) -> List[Document]:
        """
        JSONLファイルを解析
        
        Args:
            file_path: JSONLファイルパス
            
        Returns:
            List[Document]: 解析されたドキュメントリスト
        """
        documents = []
        
        try:
            with jsonlines.open(file_path, 'r') as reader:
                for i, item in enumerate(reader):
                    doc = self._process_json_item(item, file_path, i)
                    if doc:
                        documents.append(doc)
                        
        except Exception as e:
            self.logger.error(f"JSONL解析エラー {file_path}: {e}")
            raise
            
        self.logger.info(f"JSONL解析完了: {len(documents)}件のドキュメント")
        return documents
        
    def _process_json_item(self, item: Dict[str, Any], file_path: Path, index: int) -> Optional[Document]:
        """
        JSONアイテムを処理してDocumentオブジェクトを作成
        
        Args:
            item: JSONアイテム
            file_path: ファイルパス
            index: アイテムのインデックス
            
        Returns:
            Document: 処理されたドキュメント
        """
        try:
            # 様々な形式のJSONに対応
            content = self._extract_content(item)
            
            if not content:
                self.logger.warning(f"空のコンテンツをスキップ: {file_path}:{index}")
                return None
                
            # メタデータの構築
            metadata = {
                "file_path": str(file_path),
                "index": index,
                "original_keys": list(item.keys()) if isinstance(item, dict) else [],
                "item_type": type(item).__name__
            }
            
            # 元のデータから追加メタデータを抽出
            if isinstance(item, dict):
                for key, value in item.items():
                    if key not in ['text', 'content', 'instruction', 'input', 'output'] and len(str(value)) < 200:
                        metadata[f"original_{key}"] = value
                        
            return Document(
                content=content,
                metadata=metadata,
                source=str(file_path),
                doc_type="json"
            )
            
        except Exception as e:
            self.logger.error(f"JSONアイテム処理エラー {file_path}:{index}: {e}")
            return None
            
    def _extract_content(self, item: Any) -> str:
        """
        JSONアイテムからコンテンツを抽出
        
        Args:
            item: JSONアイテム
            
        Returns:
            str: 抽出されたコンテンツ
        """
        if isinstance(item, str):
            return item
            
        if isinstance(item, dict):
            # 一般的なキーから内容を探す
            content_keys = [
                'text', 'content', 'body', 'message', 'description',
                'instruction', 'output', 'response', 'answer'
            ]
            
            for key in content_keys:
                if key in item and isinstance(item[key], str) and item[key].strip():
                    return item[key].strip()
                    
            # instruction + input + output の形式
            if 'instruction' in item:
                parts = []
                if item.get('instruction'):
                    parts.append(f"指示: {item['instruction']}")
                if item.get('input'):
                    parts.append(f"入力: {item['input']}")
                if item.get('output'):
                    parts.append(f"出力: {item['output']}")
                if parts:
                    return '\n'.join(parts)
                    
            # 全てのstring値を結合
            text_parts = []
            for key, value in item.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value.strip()}")
                    
            return '\n'.join(text_parts)
            
        # その他の型は文字列に変換
        return str(item)
        
    def get_supported_extensions(self) -> List[str]:
        """
        サポートされているファイル拡張子を取得
        
        Returns:
            List[str]: サポートされている拡張子のリスト
        """
        return ['.json', '.jsonl']
        
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        ファイルが有効なJSON/JSONLファイルかを検証
        
        Args:
            file_path: ファイルパス
            
        Returns:
            bool: 有効かどうか
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
            
        if file_path.suffix.lower() not in self.get_supported_extensions():
            return False
            
        try:
            if file_path.suffix.lower() == '.jsonl':
                with jsonlines.open(file_path, 'r') as reader:
                    # 最初の行だけチェック
                    next(reader)
                    return True
            else:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    json.load(f)
                    return True
                    
        except Exception:
            return False
            
    def estimate_processing_time(self, file_path: Union[str, Path]) -> float:
        """
        処理時間の推定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            float: 推定処理時間（秒）
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return 0.0
            
        # ファイルサイズベースの推定
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # JSON: 約1MB/秒、JSONL: 約2MB/秒の処理速度を想定
        if file_path.suffix.lower() == '.jsonl':
            return file_size_mb / 2.0
        else:
            return file_size_mb / 1.0