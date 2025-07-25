"""
Professional Statistics Suite - Distribution Package
配布とビルドシステムのパッケージ
"""

# 配布関連モジュールのインポート
from src.distribution.booth_build_system import (
    BoothBuilder,
    main as booth_build_main
)

from src.distribution.booth_deployment_automation import (
    BoothDeploymentManager,
    main as booth_deployment_main
)

from src.distribution.booth_license_generator import (
    BoothLicenseGenerator,
    main as booth_license_main
)

from src.distribution.booth_sales_manager import (
    BoothSalesManager,
    main as booth_sales_main
)

from src.distribution.exe_builder_system import (
    ExeBuilderSystem,
    main as exe_builder_main
)

from src.distribution.build_exe_auto import (
    check_dependencies,
    create_simple_protected_script,
    build_exe,
    main as build_exe_auto_main
)

from src.distribution.generate_booth_content import (
    main as generate_booth_content_main
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "配布とビルドシステム"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Booth Build System
    "BoothBuilder",
    "booth_build_main",
    
    # Booth Deployment Automation
    "BoothDeploymentManager",
    "booth_deployment_main",
    
    # Booth License Generator
    "BoothLicenseGenerator",
    "booth_license_main",
    
    # Booth Sales Manager
    "BoothSalesManager",
    "booth_sales_main",
    
    # Exe Builder System
    "ExeBuilderSystem",
    "exe_builder_main",
    
    # Build Exe Auto
    "check_dependencies",
    "create_simple_protected_script",
    "build_exe",
    "build_exe_auto_main",
    
    # Generate Booth Content
    "generate_booth_content_main",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

