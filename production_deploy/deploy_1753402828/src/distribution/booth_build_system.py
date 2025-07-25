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