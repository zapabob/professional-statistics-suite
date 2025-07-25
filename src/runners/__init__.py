"""
Professional Statistics Suite - Runners Package
アプリケーション実行とデモのパッケージ
"""

# 実行関連モジュールのインポート
from src.runners.run_professional_gui import (
    check_dependencies,
    install_dependencies,
    show_welcome_message,
    main as run_professional_gui_main
)

from src.runners.run_web_dashboard import (
    check_dependencies as check_web_dependencies,
    get_optimal_settings,
    run_web_dashboard,
    main as run_web_dashboard_main
)

from src.runners.run_unified_ai_landing import (
    check_dependencies as check_ai_dependencies,
    check_env_variables,
    load_env_file,
    check_ai_modules,
    check_statistical_modules,
    check_assumption_modules,
    check_gui_modules,
    setup_environment as setup_ai_environment,
    main as run_unified_ai_landing_main
)

from src.runners.run_kiro_gui import (
    check_dependencies as check_kiro_dependencies,
    install_dependencies as install_kiro_dependencies,
    show_welcome_message as show_kiro_welcome_message,
    main as run_kiro_gui_main
)

from src.runners.launch_booth_gui import (
    check_dependencies as check_booth_dependencies,
    install_missing_packages,
    main as launch_booth_gui_main
)

from src.runners.statistical_advisor_demo import (
    create_demo_datasets,
    demo_data_characteristics_analysis,
    demo_method_suggestions,
    demo_expertise_level_adaptation,
    demo_data_quality_impact,
    demo_computation_time_estimation,
    main as statistical_advisor_demo_main
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "アプリケーション実行とデモ"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Professional GUI Runner
    "check_dependencies",
    "install_dependencies",
    "show_welcome_message",
    "run_professional_gui_main",
    
    # Web Dashboard Runner
    "check_web_dependencies",
    "get_optimal_settings",
    "run_web_dashboard",
    "run_web_dashboard_main",
    
    # Unified AI Landing Runner
    "check_ai_dependencies",
    "check_env_variables",
    "load_env_file",
    "check_ai_modules",
    "check_statistical_modules",
    "check_assumption_modules",
    "check_gui_modules",
    "setup_ai_environment",
    "run_unified_ai_landing_main",
    
    # Kiro GUI Runner
    "check_kiro_dependencies",
    "install_kiro_dependencies",
    "show_kiro_welcome_message",
    "run_kiro_gui_main",
    
    # Booth GUI Runner
    "check_booth_dependencies",
    "install_missing_packages",
    "launch_booth_gui_main",
    
    # Statistical Advisor Demo
    "create_demo_datasets",
    "demo_data_characteristics_analysis",
    "demo_method_suggestions",
    "demo_expertise_level_adaptation",
    "demo_data_quality_impact",
    "demo_computation_time_estimation",
    "statistical_advisor_demo_main",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

