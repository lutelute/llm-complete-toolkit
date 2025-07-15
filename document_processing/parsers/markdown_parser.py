import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import markdown
from markdown.extensions import toc, codehilite, tables


class MarkdownParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.md = markdown.Markdown(
            extensions=['toc', 'codehilite', 'tables', 'fenced_code'],
            extension_configs={
                'toc': {'anchorlink': True}
            }
        )
    
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Markdownファイルを解析してテキストを抽出
        
        Args:
            file_path: Markdownファイルのパス
            
        Returns:
            抽出されたテキストデータのリスト
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # メタデータの抽出（フロントマターがある場合）
            metadata = self._extract_frontmatter(content)
            if metadata['frontmatter']:
                content = metadata['content']
            
            # HTMLに変換
            html_content = self.md.convert(content)
            
            # プレーンテキストを抽出
            plain_text = self._html_to_text(html_content)
            
            # セクションごとに分割
            sections = self._split_by_sections(content)
            
            # 全体のドキュメント
            documents.append({
                'source': str(file_path),
                'type': 'markdown',
                'title': metadata['frontmatter'].get('title', file_path.stem),
                'content': plain_text,
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'frontmatter': metadata['frontmatter'],
                    'section_count': len(sections)
                }
            })
            
            # セクションごとのドキュメント
            for i, section in enumerate(sections, 1):
                if section['content'].strip():
                    section_html = self.md.convert(section['content'])
                    section_text = self._html_to_text(section_html)
                    
                    documents.append({
                        'source': str(file_path),
                        'type': 'markdown_section',
                        'title': section['title'] or f"{file_path.stem} - Section {i}",
                        'content': section_text,
                        'metadata': {
                            'section_number': i,
                            'total_sections': len(sections),
                            'heading_level': section['level']
                        }
                    })
        
        except Exception as e:
            self.logger.error(f"Markdownファイルの解析に失敗: {e}")
            raise
        
        return documents
    
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        フロントマターを抽出
        """
        frontmatter = {}
        remaining_content = content
        
        # YAML形式のフロントマターをチェック
        if content.startswith('---\n'):
            try:
                parts = content.split('---\n', 2)
                if len(parts) >= 3:
                    frontmatter_text = parts[1]
                    remaining_content = parts[2]
                    
                    # 簡単なYAMLパースing (本格的にはpyyamlを使用)
                    for line in frontmatter_text.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            frontmatter[key.strip()] = value.strip().strip('"\'')
            except Exception as e:
                self.logger.warning(f"フロントマターの解析に失敗: {e}")
        
        return {
            'frontmatter': frontmatter,
            'content': remaining_content
        }
    
    def _split_by_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        見出しによってセクションに分割
        """
        sections = []
        lines = content.split('\n')
        current_section = {
            'title': None,
            'level': 0,
            'content': ''
        }
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                # 前のセクションを保存
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # 新しいセクションを開始
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                current_section = {
                    'title': title,
                    'level': level,
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # 最後のセクションを追加
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _html_to_text(self, html: str) -> str:
        """
        HTMLからプレーンテキストを抽出
        """
        # HTMLタグを除去
        text = re.sub(r'<[^>]+>', '', html)
        
        # HTMLエンティティをデコード
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # 余分な空白を整理
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()