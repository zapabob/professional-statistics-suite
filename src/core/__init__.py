"""
Professional Statistics Suite - Core Package
コア機能と設定管理のパッケージ
"""

# コア関連モジュールのインポート
from src.core.main import (
    StatisticalAnalysisGUI,
    SessionManager,
    MLAnalysisWindow,
    DeepLearningWindow,
    VariableSelectionWindow,
    AIAnalysisWindow,
    AdvancedVisualizer,
    DataPreprocessor
)

from src.core.config import (
    PerformanceProfile,
    SPSSGradeConfig,
    HardwareDetector,
    AIConfig
)

from src.core.professional_utils import (
    ProfessionalLogger,
    PerformanceMonitor,
    ExceptionHandler,
    SecurityManager,
    DatabaseManager
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "コア機能と設定管理"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Main Application
    "StatisticalAnalysisGUI",
    "SessionManager",
    "MLAnalysisWindow",
    "DeepLearningWindow",
    "VariableSelectionWindow",
    "AIAnalysisWindow",
    "AdvancedVisualizer",
    "DataPreprocessor",
    
    # Configuration
    "PerformanceProfile",
    "SPSSGradeConfig",
    "HardwareDetector",
    "AIConfig",
    
    # Utilities
    "ProfessionalLogger",
    "PerformanceMonitor",
    "ExceptionHandler",
    "SecurityManager",
    "DatabaseManager",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

