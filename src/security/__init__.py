"""
Professional Statistics Suite - Security Package
セキュリティとライセンス管理のパッケージ
"""

# セキュリティ関連モジュールのインポート
from .audit_compliance_system import (
    AuditTrailManager,
    ComplianceChecker,
    DataPrivacyManager
)

from src.security.booth_protection import (
    AntiDebugProtection,
    LicenseManager,
    CodeObfuscator,
    IntegrityChecker,
    BoothProtectionSystem
)

from src.security.trial_license_system import (
    TrialLicenseManager
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "セキュリティとライセンス管理"

# 利用可能なクラスと関数のリスト
__all__ = [
    # Audit Compliance System
    "AuditTrailManager",
    "ComplianceChecker",
    "DataPrivacyManager",
    
    # Booth Protection
    "AntiDebugProtection",
    "LicenseManager",
    "CodeObfuscator",
    "IntegrityChecker",
    "BoothProtectionSystem",
    
    # Trial License System
    "TrialLicenseManager",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

