"""
Professional Statistics Suite - Data Package
データ処理とサンプルデータのパッケージ
"""

# データ関連モジュールのインポート
from src.data.data_preprocessing import (
    DataPreprocessor
)

from src.data.sample_data import (
    create_sample_datasets
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "データ処理とサンプルデータ"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Data Preprocessing
    "DataPreprocessor",
    
    # Sample Data
    "create_sample_datasets",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

