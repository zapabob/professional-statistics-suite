"""
Professional Statistics Suite - GUI Package
グラフィカルユーザーインターフェースのパッケージ
"""

# GUI関連モジュールのインポート
from src.gui.professional_statistics_gui import (
    ProfessionalStatisticsGUI
)

from src.gui.gui_responsiveness_optimizer import (
    ResponsivenessMetrics,
    GUIResponsivenessOptimizer,
    ResponsivenessTestSuite
)

from src.gui.HAD_Statistics_GUI import (
    HADStatisticsGUI
)

from src.gui.unified_ai_landing_gui import (
    UnifiedAILandingGUI
)

from src.gui.kiro_integrated_gui import (
    KiroIntegratedGUI
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "グラフィカルユーザーインターフェース"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Professional Statistics GUI
    "ProfessionalStatisticsGUI",
    
    # GUI Responsiveness Optimizer
    "ResponsivenessMetrics",
    "GUIResponsivenessOptimizer",
    "ResponsivenessTestSuite",
    
    # HAD Statistics GUI
    "HADStatisticsGUI",
    
    # Unified AI Landing GUI
    "UnifiedAILandingGUI",
    
    # Kiro Integrated GUI
    "KiroIntegratedGUI",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

