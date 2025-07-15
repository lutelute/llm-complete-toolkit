#!/usr/bin/env python3
"""
テキストファイルパーサー
テキストファイルからデータを抽出し、学習データ形式に変換する
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass
import chardet


@dataclass
class Document:
    """ドキュメントを表すデータクラス"""
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str = "text"


class TextParser:
    """テキストファイルパーサー"""
    
    def __init__(self, encoding: str = "utf-8", chunk_size: int = 1000):
        """
        Args:
            encoding: ファイルエンコーディング
            chunk_size: チャンクサイズ（文字数）
        """
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
    def parse(self, file_path: Union[str, Path]) -> List[Document]:
        """
        テキストファイルを解析
        
        Args:
            file_path: ファイルパス
            
        Returns:
            List[Document]: 解析されたドキュメントリスト
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
        self.logger.info(f"テキストファイル解析開始: {file_path}")
        
        try:
            # エンコーディングの自動検出
            encoding = self._detect_encoding(file_path)
            
            # ファイルを読み込み
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                
            if not content.strip():
                self.logger.warning(f"空のファイルです: {file_path}")
                return []
                
            # ドキュメントの分割
            documents = self._split_document(content, file_path, encoding)
            
            self.logger.info(f"テキスト解析完了: {len(documents)}件のドキュメント")
            return documents
            
        except Exception as e:
            self.logger.error(f"テキスト解析エラー {file_path}: {e}")
            raise
            
    def _detect_encoding(self, file_path: Path) -> str:
        """
        ファイルのエンコーディングを検出
        
        Args:
            file_path: ファイルパス
            
        Returns:
            str: 検出されたエンコーディング
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            
            self.logger.info(f"エンコーディング検出: {detected_encoding} (信頼度: {confidence:.2f})")
            
            # 信頼度が低い場合はデフォルトエンコーディングを使用
            if confidence < 0.7:
                self.logger.warning(f"信頼度が低いため、デフォルトエンコーディング({self.encoding})を使用")
                return self.encoding
                
            return detected_encoding or self.encoding
            
        except Exception as e:
            self.logger.warning(f"エンコーディング検出失敗: {e}")
            return self.encoding
            
    def _split_document(self, content: str, file_path: Path, encoding: str) -> List[Document]:
        """
        ドキュメントを適切なサイズに分割
        
        Args:
            content: ドキュメントの内容
            file_path: ファイルパス
            encoding: エンコーディング
            
        Returns:
            List[Document]: 分割されたドキュメントリスト
        """
        documents = []
        
        # 行単位で分割
        lines = content.split('\n')
        
        # 段落単位で分割を試行
        paragraphs = self._split_into_paragraphs(lines)
        
        if not paragraphs:
            # 段落分割できない場合は文字数で分割
            paragraphs = self._split_by_length(content)
            
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # メタデータの構築
            metadata = {
                "file_path": str(file_path),
                "encoding": encoding,
                "chunk_index": i,
                "total_chunks": len(paragraphs),
                "file_size": file_path.stat().st_size,
                "character_count": len(paragraph),
                "line_count": paragraph.count('\n') + 1
            }
            
            # ファイル名から追加情報を抽出
            self._extract_file_metadata(file_path, metadata)
            
            document = Document(
                content=paragraph.strip(),
                metadata=metadata,
                source=str(file_path),
                doc_type="text"
            )
            
            documents.append(document)
            
        return documents
        
    def _split_into_paragraphs(self, lines: List[str]) -> List[str]:
        """
        行を段落に分割
        
        Args:
            lines: 行のリスト
            
        Returns:
            List[str]: 段落のリスト
        """
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # 空行で段落を区切る
                if current_paragraph:
                    paragraph_text = '\n'.join(current_paragraph)
                    if len(paragraph_text) > 50:  # 最小長さチェック
                        paragraphs.append(paragraph_text)
                    current_paragraph = []
            else:
                current_paragraph.append(line)
                
                # 段落が長すぎる場合は分割
                if len('\n'.join(current_paragraph)) > self.chunk_size:
                    paragraph_text = '\n'.join(current_paragraph)
                    paragraphs.append(paragraph_text)
                    current_paragraph = []
                    
        # 最後の段落を追加
        if current_paragraph:
            paragraph_text = '\n'.join(current_paragraph)
            if len(paragraph_text) > 50:
                paragraphs.append(paragraph_text)
                
        return paragraphs
        
    def _split_by_length(self, content: str) -> List[str]:
        """
        文字数で分割
        
        Args:
            content: 分割するテキスト
            
        Returns:
            List[str]: 分割されたテキストのリスト
        """
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # 単語境界で分割を試行
            if end < len(content):
                # 句読点で分割
                for delimiter in ['.', '。', '!', '！', '?', '？', '\n']:
                    delimiter_pos = content.rfind(delimiter, start, end)
                    if delimiter_pos > start:
                        end = delimiter_pos + 1
                        break
                        
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end
            
        return chunks
        
    def _extract_file_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """
        ファイルパスからメタデータを抽出
        
        Args:
            file_path: ファイルパス
            metadata: メタデータ辞書（更新される）
        """
        metadata["file_name"] = file_path.name
        metadata["file_stem"] = file_path.stem
        metadata["file_extension"] = file_path.suffix
        metadata["parent_directory"] = file_path.parent.name
        
        # ファイル名から情報を推定
        file_name = file_path.stem.lower()
        
        # ドキュメントタイプの推定
        if any(word in file_name for word in ['readme', 'read_me', 'ドキュメント', 'document']):
            metadata["document_type"] = "documentation"
        elif any(word in file_name for word in ['tutorial', 'guide', 'ガイド', 'チュートリアル']):
            metadata["document_type"] = "tutorial"
        elif any(word in file_name for word in ['note', 'notes', 'メモ', 'ノート']):
            metadata["document_type"] = "note"
        elif any(word in file_name for word in ['log', 'ログ']):
            metadata["document_type"] = "log"
        elif any(word in file_name for word in ['config', 'configuration', '設定']):
            metadata["document_type"] = "configuration"
        else:
            metadata["document_type"] = "general"
            
    def get_supported_extensions(self) -> List[str]:
        """
        サポートされているファイル拡張子を取得
        
        Returns:
            List[str]: サポートされている拡張子のリスト
        """
        return ['.txt', '.text', '.log', '.csv', '.tsv']
        
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        ファイルが有効なテキストファイルかを検証
        
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
            # エンコーディングの検出を試行
            encoding = self._detect_encoding(file_path)
            
            # ファイルの読み込みを試行
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(100)  # 最初の100文字を読んでみる
                
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
        
        # 約5MB/秒の処理速度を想定
        return file_size_mb / 5.0