import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import jsonlines


class LMStudioConverter:
    def __init__(self, chunk_size: int = 512, output_format: str = 'jsonl'):
        """
        LM Studio用のデータコンバーター
        
        Args:
            chunk_size: テキストチャンクの最大サイズ
            output_format: 出力フォーマット ('jsonl' または 'text')
        """
        self.chunk_size = chunk_size
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)
    
    def convert(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ドキュメントをLM Studio用の形式に変換
        
        Args:
            documents: 解析済みドキュメントのリスト
            
        Returns:
            変換されたデータのリスト
        """
        converted_data = []
        
        for doc in documents:
            if not doc.get('content'):
                continue
                
            # テキストをチャンクに分割
            chunks = self._split_text(doc['content'], self.chunk_size)
            
            for i, chunk in enumerate(chunks):
                if self.output_format == 'jsonl':
                    # JSONL形式（LM Studio推奨）
                    converted_item = {
                        'text': chunk,
                        'source': doc['source'],
                        'title': doc['title'],
                        'type': doc['type'],
                        'chunk_id': i + 1,
                        'total_chunks': len(chunks),
                        'metadata': doc.get('metadata', {})
                    }
                else:
                    # プレーンテキスト形式
                    converted_item = {
                        'content': chunk,
                        'metadata': {
                            'source': doc['source'],
                            'title': doc['title'],
                            'type': doc['type'],
                            'chunk_id': i + 1,
                            'total_chunks': len(chunks)
                        }
                    }
                
                converted_data.append(converted_item)
        
        self.logger.info(f"変換完了: {len(converted_data)}チャンクを生成")
        return converted_data
    
    def convert_for_instruction_tuning(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        インストラクションチューニング用の形式に変換
        """
        converted_data = []
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue
            
            # 簡単な質問応答ペアを生成
            chunks = self._split_text(content, self.chunk_size)
            
            for chunk in chunks:
                # ドキュメントの内容に基づく質問応答を生成
                qa_pairs = self._generate_qa_pairs(chunk, doc)
                converted_data.extend(qa_pairs)
        
        return converted_data
    
    def _generate_qa_pairs(self, text: str, doc_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        テキストから質問応答ペアを生成
        """
        qa_pairs = []
        
        # 基本的な質問パターン
        questions = [
            f"{doc_info['title']}について説明してください。",
            f"この文書の内容を要約してください。",
            f"{doc_info['title']}の主要なポイントは何ですか？"
        ]
        
        for question in questions:
            qa_pair = {
                "instruction": question,
                "input": "",
                "output": text[:500] + "..." if len(text) > 500 else text,
                "metadata": {
                    "source": doc_info['source'],
                    "type": doc_info['type']
                }
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        テキストを指定されたサイズのチャンクに分割
        """
        if not text:
            return []
        
        # 文章境界で分割を試行
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # チャンクサイズを超える場合は新しいチャンクを開始
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # 最後のチャンクを追加
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        テキストを文単位で分割
        """
        import re
        
        # 日本語と英語の文境界を検出
        sentence_endings = r'[.!?。！？]'
        sentences = re.split(f'({sentence_endings})', text)
        
        # 分割結果を再構成
        result = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence.strip():
                result.append(sentence.strip())
        
        return result
    
    def save(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """
        データをファイルに保存
        
        Args:
            data: 保存するデータ
            output_path: 出力ファイルパス
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == 'jsonl':
            with jsonlines.open(output_path, mode='w') as writer:
                for item in data:
                    writer.write(item)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(item['content'] + '\n\n---\n\n')
        
        self.logger.info(f"データを保存しました: {output_path}")
    
    def save_instruction_format(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """
        インストラクションチューニング形式でデータを保存
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_path, mode='w') as writer:
            for item in data:
                writer.write(item)
        
        self.logger.info(f"インストラクション形式でデータを保存: {output_path}")