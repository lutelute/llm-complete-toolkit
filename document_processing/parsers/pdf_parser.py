import logging
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import pdfplumber


class PDFParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        PDFファイルを解析してテキストを抽出
        
        Args:
            file_path: PDFファイルのパス
            
        Returns:
            抽出されたテキストデータのリスト
        """
        documents = []
        
        try:
            # pdfplumberを使用してテキストを抽出
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        page_text = self._clean_text(page_text)
                        page_texts.append({
                            'page_number': page_num,
                            'text': page_text
                        })
                        full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                
                # 全体のドキュメント
                documents.append({
                    'source': str(file_path),
                    'type': 'pdf',
                    'title': file_path.stem,
                    'content': full_text.strip(),
                    'metadata': {
                        'page_count': len(pdf.pages),
                        'file_size': file_path.stat().st_size
                    }
                })
                
                # ページごとのドキュメント
                for page_data in page_texts:
                    documents.append({
                        'source': str(file_path),
                        'type': 'pdf_page',
                        'title': f"{file_path.stem} - Page {page_data['page_number']}",
                        'content': page_data['text'],
                        'metadata': {
                            'page_number': page_data['page_number'],
                            'total_pages': len(pdf.pages)
                        }
                    })
        
        except Exception as e:
            self.logger.error(f"PDFplumberでの解析に失敗: {e}")
            # フォールバックとしてPyPDF2を使用
            documents = self._parse_with_pypdf2(file_path)
        
        return documents
    
    def _parse_with_pypdf2(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        PyPDF2を使用したフォールバック解析
        """
        documents = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        page_text = self._clean_text(page_text)
                        full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                
                documents.append({
                    'source': str(file_path),
                    'type': 'pdf',
                    'title': file_path.stem,
                    'content': full_text.strip(),
                    'metadata': {
                        'page_count': len(pdf_reader.pages),
                        'file_size': file_path.stat().st_size,
                        'parser': 'PyPDF2'
                    }
                })
        
        except Exception as e:
            self.logger.error(f"PyPDF2での解析にも失敗: {e}")
            raise
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        テキストのクリーニング
        """
        if not text:
            return ""
        
        # 余分な空白や改行を整理
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)