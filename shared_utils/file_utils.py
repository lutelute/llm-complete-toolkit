import logging
from pathlib import Path
from typing import List, Set


def get_files_by_extension(directory: Path, extension: str) -> List[Path]:
    """
    指定されたディレクトリから特定の拡張子のファイルを取得
    
    Args:
        directory: 検索するディレクトリ
        extension: ファイル拡張子（例: '.pdf', '.md'）
        
    Returns:
        ファイルパスのリスト
    """
    if not directory.exists():
        raise FileNotFoundError(f"ディレクトリが存在しません: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"指定されたパスはディレクトリではありません: {directory}")
    
    # 拡張子が.で始まっていない場合は追加
    if not extension.startswith('.'):
        extension = '.' + extension
    
    files = []
    
    # 再帰的にファイルを検索
    for file_path in directory.rglob(f'*{extension}'):
        if file_path.is_file():
            files.append(file_path)
    
    return sorted(files)


def get_supported_files(directory: Path, supported_extensions: Set[str] = None) -> List[Path]:
    """
    サポートされている拡張子のファイルを全て取得
    
    Args:
        directory: 検索するディレクトリ
        supported_extensions: サポートされる拡張子のセット
        
    Returns:
        ファイルパスのリスト
    """
    if supported_extensions is None:
        supported_extensions = {'.pdf', '.md', '.markdown', '.txt'}
    
    all_files = []
    
    for ext in supported_extensions:
        files = get_files_by_extension(directory, ext)
        all_files.extend(files)
    
    return sorted(all_files)


def validate_input_directory(directory: Path) -> bool:
    """
    入力ディレクトリの検証
    
    Args:
        directory: 検証するディレクトリ
        
    Returns:
        検証結果
    """
    logger = logging.getLogger(__name__)
    
    if not directory.exists():
        logger.error(f"入力ディレクトリが存在しません: {directory}")
        return False
    
    if not directory.is_dir():
        logger.error(f"指定されたパスはディレクトリではありません: {directory}")
        return False
    
    # サポートされているファイルがあるかチェック
    supported_files = get_supported_files(directory)
    if not supported_files:
        logger.warning(f"サポートされているファイルが見つかりません: {directory}")
        return False
    
    logger.info(f"入力ディレクトリの検証成功: {len(supported_files)}個のファイルを発見")
    return True


def create_output_directory(directory: Path) -> bool:
    """
    出力ディレクトリの作成
    
    Args:
        directory: 作成するディレクトリ
        
    Returns:
        作成結果
    """
    logger = logging.getLogger(__name__)
    
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"出力ディレクトリを作成/確認しました: {directory}")
        return True
    except Exception as e:
        logger.error(f"出力ディレクトリの作成に失敗: {e}")
        return False


def get_file_info(file_path: Path) -> dict:
    """
    ファイルの基本情報を取得
    
    Args:
        file_path: ファイルパス
        
    Returns:
        ファイル情報の辞書
    """
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが存在しません: {file_path}")
    
    stat = file_path.stat()
    
    return {
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'size': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified_time': stat.st_mtime,
        'absolute_path': str(file_path.absolute())
    }