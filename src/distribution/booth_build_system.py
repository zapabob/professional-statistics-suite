#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Booth Build System
商用版ビルドシステム - PyInstaller + 難読化 + 保護機能
"""

import os
import sys
import shutil
import subprocess
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Any

class BoothBuilder:
    """Booth版ビルドシステム"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
    
    def build_booth_version(self, entry_point: str, icon_path: str = None, additional_files: List[str] = None) -> Dict[str, Any]:
        """Booth版ビルド実行"""
        try:
            # PyInstallerでビルド
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                "--name=StatisticsSuite_Booth",
                entry_point
            ]
            
            if icon_path and os.path.exists(icon_path):
                cmd.extend(["--icon", icon_path])
            
            # 追加ファイル
            if additional_files:
                for file_path in additional_files:
                    if os.path.exists(file_path):
                        cmd.extend(["--add-data", f"{file_path};."])
            
            # ビルド実行
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            
            # パッケージ作成
            package_path = self._create_booth_package()
            
            return {
                "success": True,
                "package_path": package_path,
                "build_dir": str(self.dist_dir)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_booth_package(self) -> str:
        """Booth版パッケージ作成"""
        package_name = f"StatisticsSuite_Booth_v1.0.0.zip"
        package_path = self.project_root / package_name
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 実行ファイル
            for exe_file in self.dist_dir.glob("*.exe"):
                zf.write(exe_file, exe_file.name)
            
            # ドキュメント
            docs = ["README.md", "LICENSE"]
            for doc in docs:
                doc_path = self.project_root / doc
                if doc_path.exists():
                    zf.write(doc_path, doc)
            
            # サンプルデータ（あれば）
            sample_data = self.project_root / "sample_data"
            if sample_data.exists():
                for file_path in sample_data.rglob("*"):
                    if file_path.is_file():
                        arc_path = file_path.relative_to(self.project_root)
                        zf.write(file_path, str(arc_path))
            
            # ライセンス認証ガイド
            license_guide = """
# ライセンス認証ガイド

## 初回起動時
1. アプリケーションを実行
2. ライセンス認証URLが表示されます
3. URLにアクセスしてライセンスを購入・認証
4. 認証完了後、アプリケーションが使用可能になります

## サポート
- メール: support@your-domain.com
- Booth: https://your-booth-shop.booth.pm/
"""
            
            zf.writestr("ライセンス認証ガイド.txt", license_guide.encode('utf-8'))
        
        return str(package_path)

# ビルドスクリプト
def main():
    """Booth版ビルド実行"""
    builder = BoothBuilder(".")
    
    result = builder.build_booth_version(
        entry_point="HAD_Statistics_GUI.py",
        icon_path="icon.ico",  # アイコンファイルがある場合
        additional_files=[
            "requirements.txt",
            "config.py",
            "templates/",
            "reports/"
        ]
    )
    
    if result["success"]:
        print(f"✅ ビルド成功: {result['package_path']}")
    else:
        print(f"❌ ビルド失敗: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main() 