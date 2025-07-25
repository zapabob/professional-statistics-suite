#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXE Builder System with License Protection
ライセンス保護付きEXE化システム
"""

import shutil
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class ExeBuilderSystem:
    """EXE化システム"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.build_dir = self.project_root / "build_exe"
        self.dist_dir = self.project_root / "dist"
        self.temp_dir = self.project_root / "temp_build"
        
        # ビルド設定
        self.build_config = {
            "python_scripts": [
                "main.py",
                "advanced_statistics.py", 
                "advanced_visualization.py",
                "ai_integration.py",
                "bayesian_analysis.py",
                "data_preprocessing.py",
                "HAD_Statistics_GUI.py",
                "ml_pipeline_automation.py",
                "parallel_optimization.py",
                "professional_reports.py",
                "professional_utils.py",
                "run_web_dashboard.py",
                "sample_data.py",
                "survival_analysis.py",
                "web_dashboard.py",
                "booth_license_gui.py",
                "booth_license_simple_gui.py"
            ],
            "icon_file": "icon.ico",
            "company_name": "Professional Statistics Suite",
            "file_version": "1.0.0.0",
            "product_version": "1.0.0",
            "copyright": f"Copyright (C) {datetime.now().year} Professional Statistics Suite",
            "description": "Professional Statistics Analysis Suite"
        }
        
    def create_protected_entry_point(self, script_path: str, app_name: str) -> str:
        """ライセンス保護エントリーポイント作成"""
        
        protected_code = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Protected Entry Point for {app_name}
ライセンス保護エントリーポイント
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from trial_license_system import TrialLicenseManager, require_trial_or_license
    from booth_protection import BoothProtectionSystem, anti_debug
    
    # 保護システム初期化
    protection_system = BoothProtectionSystem()
    trial_manager = TrialLicenseManager("{app_name}")
    
    @anti_debug
    @require_trial_or_license("{app_name}")
    def main():
        """保護されたメイン関数"""
        try:
            # ライセンス状態表示
            license_status = trial_manager.check_license()
            print(f"🔒 {{license_status['message']}}")
            print("=" * 50)
            
            # 保護システム初期化
            protection_result = protection_system.initialize_protection()
            if not protection_result.get("success", False):
                print(f"❌ 保護システムエラー: {{protection_result.get('error')}}")
                sys.exit(1)
            
            # オリジナルスクリプト実行
            original_script = Path(__file__).parent / "{Path(script_path).name}"
            
            if not original_script.exists():
                print(f"❌ メインスクリプトが見つかりません: {{original_script}}")
                sys.exit(1)
            
            # オリジナルスクリプトの内容を実行
            with open(original_script, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # グローバル名前空間でスクリプト実行
            exec(script_content, {{'__name__': '__main__', '__file__': str(original_script)}})
            
        except SystemExit:
            raise
        except Exception as e:
            print(f"❌ アプリケーションエラー: {{e}}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ 必要なモジュールが見つかりません: {{e}}")
    print("ライセンスシステムが正しくインストールされているか確認してください")
    sys.exit(1)
except Exception as e:
    print(f"❌ 初期化エラー: {{e}}")
    sys.exit(1)
'''
        
        return protected_code
    
    def create_pyinstaller_spec(self, script_path: str, app_name: str, exe_name: str) -> str:
        """PyInstallerスペックファイル作成"""
        
        current_dir_str = str(self.project_root)
        
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for {app_name}
"""

import sys
from pathlib import Path

# ビルド設定
block_cipher = None
current_dir = Path("{current_dir_str}")

# データファイル
datas = [
    ('trial_license_system.py', '.'),
    ('booth_protection.py', '.'),
    ('config.py', '.'),
    ('professional_utils.py', '.'),
    ('templates', 'templates'),
    ('checkpoints', 'checkpoints'),
    ('reports', 'reports'),
    ('logs', 'logs')
]

# 隠蔽ファイル (--hidden-import)
hiddenimports = [
    'trial_license_system',
    'booth_protection', 
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'scipy',
    'sklearn',
    'tensorflow',
    'torch',
    'plotly',
    'dash',
    'qrcode',
    'PIL',
    'cryptography',
    'psutil',
    'tqdm',
    'joblib',
    'sqlite3',
    'json',
    'pickle',
    'base64',
    'hashlib',
    'uuid',
    'platform',
    'winreg',
    'ctypes'
]

# バイナリ収集
binaries = []

# 解析フェーズ
a = Analysis(
    ['{script_path}'],
    pathex=[str(current_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PYZ作成
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 実行ファイル作成
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{exe_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # GUIアプリの場合はFalseに変更
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='{current_dir_str}/version_info.txt',
    icon='{current_dir_str}/icon.ico' if Path('{current_dir_str}/icon.ico').exists() else None
)
'''
        
        return spec_content
    
    def create_version_info(self, app_name: str) -> str:
        """バージョン情報ファイル作成"""
        
        version_info = f'''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
# filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
# Set not needed items to zero 0.
filevers=(1,0,0,0),
prodvers=(1,0,0,0),
# Contains a bitmask that specifies the valid bits 'flags'r
mask=0x3f,
# Contains a bitmask that specifies the Boolean attributes of the file.
flags=0x0,
# The operating system for which this file was designed.
OS=0x40004,
# The general type of file.
fileType=0x1,
# The function of the file.
subtype=0x0,
# Creation date and time stamp.
date=(0, 0)
),
  kids=[
StringFileInfo(
  [
  StringTable(
    u'041104B0',
    [StringStruct(u'CompanyName', u'{self.build_config["company_name"]}'),
    StringStruct(u'FileDescription', u'{app_name}'),
    StringStruct(u'FileVersion', u'{self.build_config["file_version"]}'),
    StringStruct(u'InternalName', u'{app_name}'),
    StringStruct(u'LegalCopyright', u'{self.build_config["copyright"]}'),
    StringStruct(u'OriginalFilename', u'{app_name}.exe'),
    StringStruct(u'ProductName', u'{app_name}'),
    StringStruct(u'ProductVersion', u'{self.build_config["product_version"]}')])
  ]), 
VarFileInfo([VarStruct(u'Translation', [1041, 1200])])
  ]
)
'''
        return version_info
    
    def build_single_exe(self, script_path: str, app_name: str, exe_name: Optional[str] = None) -> bool:
        """単一EXEファイルビルド（PyInstallerコマンド生成）"""
        try:
            if exe_name is None:
                exe_name = Path(script_path).stem
            
            print(f"🔨 Building {app_name} ({script_path}) -> {exe_name}.exe")
            
            # 一時ディレクトリ作成
            self.temp_dir.mkdir(exist_ok=True)
            
            # 保護されたエントリーポイント作成
            protected_entry = self.temp_dir / f"protected_{exe_name}.py"
            protected_code = self.create_protected_entry_point(script_path, app_name)
            
            with open(protected_entry, 'w', encoding='utf-8') as f:
                f.write(protected_code)
            
            # PyInstallerコマンドを生成して表示
            build_cmd = [
                "python", "-m", "PyInstaller",
                "--onefile",
                "--noconsole" if "gui" in script_path.lower() else "--console",
                "--clean",
                "--noconfirm",
                f"--distpath={self.dist_dir}",
                f"--workpath={self.build_dir}",
                "--add-data", "trial_license_system.py;.",
                "--add-data", "booth_protection.py;.",
                "--hidden-import", "trial_license_system",
                "--hidden-import", "booth_protection",
                "--hidden-import", "tkinter",
                "--hidden-import", "cryptography",
                f"--name={exe_name}",
                str(protected_entry)
            ]
            
            print("📦 PyInstallerコマンド:")
            print(" ".join(build_cmd))
            print()
            
            return True
                
        except Exception as e:
            print(f"❌ ビルドコマンド生成エラー: {e}")
            return False
    
    def build_all_executables(self) -> Dict[str, bool]:
        """全実行ファイルビルド"""
        print("🚀 Professional Statistics Suite EXE Builder")
        print("=" * 60)
        
        # ビルドディレクトリ作成
        self.dist_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)
        
        # アプリケーション定義
        applications = {
            "main.py": "Professional Statistics Suite",
            "HAD_Statistics_GUI.py": "HAD Statistics GUI",
            "booth_license_gui.py": "BOOTH License Manager", 
            "booth_license_simple_gui.py": "BOOTH License Simple GUI",
            "run_web_dashboard.py": "Statistics Web Dashboard",
            "advanced_statistics.py": "Advanced Statistics Tool",
            "advanced_visualization.py": "Data Visualization Tool",
            "ai_integration.py": "AI Analysis Tool",
            "ml_pipeline_automation.py": "ML Pipeline Automation"
        }
        
        results = {}
        
        for script_path, app_name in applications.items():
            script_file = self.project_root / script_path
            if script_file.exists():
                exe_name = Path(script_path).stem
                success = self.build_single_exe(script_path, app_name, exe_name)
                results[script_path] = success
                print()
            else:
                print(f"⚠️  スクリプトが見つかりません: {script_path}")
                results[script_path] = False
        
        return results
    
    def create_installer_script(self) -> str:
        """インストーラースクリプト作成"""
        installer_script = '''@echo off
title Professional Statistics Suite Installer

echo ================================
echo Professional Statistics Suite
echo ================================
echo.

echo Installing executables...

if not exist "C:\\Program Files\\Professional Statistics Suite" (
    mkdir "C:\\Program Files\\Professional Statistics Suite"
)

copy *.exe "C:\\Program Files\\Professional Statistics Suite\\"
copy *.dll "C:\\Program Files\\Professional Statistics Suite\\" 2>nul

echo.
echo Creating desktop shortcuts...

echo [InternetShortcut] > "%USERPROFILE%\\Desktop\\Statistics Suite.url"
echo URL=file:///C:/Program Files/Professional Statistics Suite/main.exe >> "%USERPROFILE%\\Desktop\\Statistics Suite.url"

echo.
echo Installation completed!
echo.
echo Executables installed to: C:\\Program Files\\Professional Statistics Suite
echo Desktop shortcut created: Statistics Suite
echo.
echo Each application has a 14-day trial period.
echo Purchase a license key to unlock full functionality.
echo.
pause
'''
        return installer_script
    
    def cleanup_build_files(self):
        """ビルドファイルクリーンアップ"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
                
            print("🧹 ビルドファイルクリーンアップ完了")
            
        except Exception as e:
            print(f"⚠️  クリーンアップエラー: {e}")

def main():
    """メイン実行"""
    builder = ExeBuilderSystem()
    
    try:
        # 全EXEファイルビルドコマンド生成
        results = builder.build_all_executables()
        
        print("=" * 60)
        print("📋 次の手順:")
        print("1. 上記のPyInstallerコマンドを実行してください")
        print("2. 各EXEファイルには14日間試用期間が設定されます")
        print("3. ライセンスキーを購入することで制限が解除されます")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main() 