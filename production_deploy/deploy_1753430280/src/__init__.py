"""
Professional Statistics Suite - Production Deploy src package
本番環境用の高度な統計分析システムのメインパッケージ
"""

# メインパッケージのバージョン情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__email__ = "r.minegishi1987@gmail.com"

# 主要なモジュールのインポート
from .core import main, config
from .ai import ai_integration, contextual_retriever, gguf_model_selector
from .gui import professional_statistics_gui, gui_responsiveness_optimizer
from .statistics import (
    advanced_statistics, 
    bayesian_analysis, 
    survival_analysis,
    statistical_power_analysis,
    statistical_method_advisor,
    assumption_validator
)
from .data import data_preprocessing, sample_data
from .visualization import (
    advanced_visualization, 
    professional_reports, 
    web_dashboard
)
from .security import (
    audit_compliance_system, 
    booth_protection, 
    trial_license_system
)
from .runners import (
    run_professional_gui,
    run_web_dashboard,
    interactive_analysis_app
)
from .tests import (
    production_environment_test,
    e2e_test_automation,
    integrated_test_runner
)
from .distribution import (
    booth_build_system,
    booth_deployment_automation,
    booth_license_generator
)

# パッケージレベルの変数
PACKAGE_NAME = "Professional Statistics Suite - Production"
PACKAGE_DESCRIPTION = "本番環境用の高度な統計分析システム"
SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]
DEPLOYMENT_VERSION = "deploy_1753430280"

# 利用可能なモジュールのリスト
__all__ = [
    # Core modules
    "main",
    "config",
    
    # AI modules
    "ai_integration",
    "contextual_retriever", 
    "gguf_model_selector",
    
    # GUI modules
    "professional_statistics_gui",
    "gui_responsiveness_optimizer",
    
    # Statistics modules
    "advanced_statistics",
    "bayesian_analysis",
    "survival_analysis", 
    "statistical_power_analysis",
    "statistical_method_advisor",
    "assumption_validator",
    
    # Data modules
    "data_preprocessing",
    "sample_data",
    
    # Visualization modules
    "advanced_visualization",
    "professional_reports",
    "web_dashboard",
    
    # Security modules
    "audit_compliance_system",
    "booth_protection",
    "trial_license_system",
    
    # Runner modules
    "run_professional_gui",
    "run_web_dashboard", 
    "interactive_analysis_app",
    
    # Test modules
    "production_environment_test",
    "e2e_test_automation",
    "integrated_test_runner",
    
    # Distribution modules
    "booth_build_system",
    "booth_deployment_automation",
    "booth_license_generator",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
    "PACKAGE_NAME",
    "PACKAGE_DESCRIPTION",
    "SUPPORTED_PYTHON_VERSIONS",
    "DEPLOYMENT_VERSION"
]

