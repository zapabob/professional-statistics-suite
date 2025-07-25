"""
Professional Statistics Suite - Visualization Package
データ可視化とレポート生成のパッケージ
"""

# 可視化関連モジュールのインポート
from src.visualization.advanced_visualization import (
    AdvancedVisualizer
)

from src.visualization.professional_reports import (
    ReportGenerator
)

from src.visualization.web_dashboard import (
    MultilingualWebDashboard
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "データ可視化とレポート生成"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Advanced Visualization
    "AdvancedVisualizer",
    
    # Professional Reports
    "ReportGenerator",
    
    # Web Dashboard
    "MultilingualWebDashboard",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

