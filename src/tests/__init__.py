"""
Professional Statistics Suite - Tests Package
テストと品質保証のパッケージ
"""

# テスト関連モジュールのインポート




from src.tests.integrated_test_runner import (
    IntegratedTestRunner
)

from src.tests.cli_test_runner import (
    CLITestRunner
)

from src.tests.test_data_manager import (
    TestDataManager,
    DataGenerator,
    TestDataSet,
    DataGenerationConfig,
    DataStorage,
    DataSerializer,
    TestDataFactory,
    main as test_data_manager_main
)

from src.tests.html_report_generator import (
    HTMLReportGenerator,
    TestResult,
    CoverageData,
    PerformanceMetrics,
    ReportManager,
    main as html_report_main
)

from src.tests.coverage_analyzer import (
    CoverageAnalyzer,
    CoverageMetrics,
    ModuleCoverage,
    TestCoverageGenerator,
    main as coverage_analyzer_main
)

from src.tests.parallel_test_runner import (
    ParallelTestRunner,
    ParallelTestConfig,
    TestExecutionResult,
    PytestXdistRunner,
    main as parallel_test_main
)

from src.tests.performance_optimizer import (
    PerformanceProfiler,
    TestPerformanceMetrics,
    TestOptimizer,
    CachingManager,
    TestParallelExecutor,
    main as performance_optimizer_main
)

from src.tests.gui_button_test_automation import (
    GUIButtonTestAutomation,
    ButtonTestResult,
    ButtonStateMonitor,
    main as gui_button_test_main
)

from src.tests.simple_production_test import SimpleProductionTest

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "テストと品質保証"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Production Environment Test
    "ProductionEnvironmentTest",
    
    # E2E Test Automation
    "E2ETestAutomation",
    
    # Integrated Test Runner
    "IntegratedTestRunner",
    "TestResultAnalyzer",
    "CoverageReporter",
    
    # CLI Test Runner
    "CLITestRunner",
    "CommandLineTester",
    "InterfaceValidator",
    "UserInputTester",
    
    # Test Data Manager
    "TestDataManager",
    "DataGenerator",
    "TestDataSet",
    "DataGenerationConfig",
    "DataStorage",
    "DataSerializer",
    "TestDataFactory",
    "test_data_manager_main",
    
    # HTML Report Generator
    "HTMLReportGenerator",
    "TestResult",
    "CoverageData",
    "PerformanceMetrics",
    "ReportManager",
    "html_report_main",
    
    # Coverage Analyzer
    "CoverageAnalyzer",
    "CoverageMetrics",
    "ModuleCoverage",
    "TestCoverageGenerator",
    "coverage_analyzer_main",
    
    # Parallel Test Runner
    "ParallelTestRunner",
    "ParallelTestConfig",
    "TestExecutionResult",
    "PytestXdistRunner",
    "parallel_test_main",
    
    # Performance Optimizer
    "PerformanceProfiler",
    "TestPerformanceMetrics",
    "TestOptimizer",
    "CachingManager",
    "TestParallelExecutor",
    "performance_optimizer_main",
    
    # GUI Button Test Automation
    "GUIButtonTestAutomation",
    "ButtonTestResult",
    "ButtonStateMonitor",
    "gui_button_test_main",
    
    # Simple Production Test
    "SimpleProductionTest",
    "BasicFunctionalityTester",
    "CoreFeatureValidator",
    "EssentialTestRunner",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

