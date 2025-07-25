"""
Professional Statistics Suite - Statistics Package
高度な統計分析機能のパッケージ
"""

# 統計分析関連モジュールのインポート
from src.statistics.advanced_statistics import (
    AdvancedStatsAnalyzer
)

from src.statistics.bayesian_analysis import (
    DeepBayesianAnalyzer
)

from src.statistics.survival_analysis import (
    CompleteSurvivalAnalyzer
)

from src.statistics.statistical_power_analysis import (
    PowerAnalysisEngine
)

from src.statistics.statistical_method_advisor import (
    DataType,
    AnalysisGoal,
    StatisticalMethod,
    DataCharacteristics,
    MethodRecommendation,
    StatisticalMethodAdvisor
)

from src.statistics.assumption_validator import (
    AssumptionType,
    ViolationSeverity,
    AssumptionTest,
    ValidationResult,
    AssumptionValidator
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "高度な統計分析機能"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Advanced Statistics
    "AdvancedStatsAnalyzer",
    
    # Bayesian Analysis
    "DeepBayesianAnalyzer",
    
    # Survival Analysis
    "CompleteSurvivalAnalyzer",
    
    # Power Analysis
    "PowerAnalysisEngine",
    
    # Method Advisor
    "DataType",
    "AnalysisGoal",
    "StatisticalMethod",
    "DataCharacteristics",
    "MethodRecommendation",
    "StatisticalMethodAdvisor",
    
    # Assumption Validator
    "AssumptionType",
    "ViolationSeverity",
    "AssumptionTest",
    "ValidationResult",
    "AssumptionValidator",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

