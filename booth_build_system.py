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
import tempfile
import ast
import re

class PythonObfuscator:
    """Pythonコード難読化"""
    
    def __init__(self):
        self.variable_mapping = {}
        self.function_mapping = {}
        self.counter = 0
    
    def _generate_name(self, prefix: str = "var") -> str:
        """難読化変数名生成"""
        self.counter += 1
        return f"__{prefix}_{hex(self.counter)[2:]}__"
    
    def obfuscate_file(self, input_file: str, output_file: str):
        """ファイル難読化"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 文字列リテラルの難読化
        content = self._obfuscate_strings(content)
        
        # 変数名の難読化
        content = self._obfuscate_variables(content)
        
        # 無意味なコードの挿入
        content = self._insert_dummy_code(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _obfuscate_strings(self, content: str) -> str:
        """文字列リテラル難読化"""
        def replace_string(match):
            string_value = match.group(1)
            if len(string_value) > 3:  # 短い文字列は除外
                encoded = string_value.encode('utf-8').hex()
                return f'bytes.fromhex("{encoded}").decode("utf-8")'
            return match.group(0)
        
        # シングル/ダブルクォート文字列を置換
        content = re.sub(r'"([^"]{4,})"', replace_string, content)
        content = re.sub(r"'([^']{4,})'", replace_string, content)
        
        return content
    
    def _obfuscate_variables(self, content: str) -> str:
        """変数名難読化"""
        # 簡単な変数名パターンマッチング
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b'
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name not in ['import', 'def', 'class', 'if', 'else', 'elif', 'while', 'for', 'try', 'except', 'finally', 'with', 'return', 'yield', 'break', 'continue', 'pass', 'raise', 'assert', 'del', 'global', 'nonlocal', 'lambda', 'and', 'or', 'not', 'is', 'in']:
                if var_name not in self.variable_mapping:
                    self.variable_mapping[var_name] = self._generate_name("var")
                return self.variable_mapping[var_name]
            return var_name
        
        return re.sub(variable_pattern, replace_var, content)
    
    def _insert_dummy_code(self, content: str) -> str:
        """ダミーコード挿入"""
        dummy_imports = [
            "import hashlib",
            "import base64", 
            "import random",
            "import time",
            "import os",
        ]
        
        dummy_functions = [
            """
def __dummy_hash_func():
    return hashlib.md5(str(random.randint(1000, 9999)).encode()).hexdigest()
""",
            """
def __dummy_time_func():
    time.sleep(0.001)
    return time.time()
""",
            """
def __dummy_calc_func():
    result = 0
    for i in range(100):
        result += i * 2
    return result
""",
        ]
        
        # インポート後にダミー関数を挿入
        lines = content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_end = i + 1
        
        # ダミーコードを挿入
        dummy_code = '\n'.join(dummy_imports + dummy_functions)
        lines.insert(import_end, dummy_code)
        
        return '\n'.join(lines)

class BoothBuilder:
    """Booth版ビルダー"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.build_dir = self.project_root / "build_booth"
        self.dist_dir = self.project_root / "dist_booth"
        self.obfuscated_dir = self.project_root / "obfuscated"
        self.obfuscator = PythonObfuscator()
        
    def build_booth_version(self, 
                          entry_point: str = "main.py",
                          icon_path: str = None,
                          additional_files: List[str] = None) -> Dict[str, Any]:
        """Booth版ビルド実行"""
        try:
            print("🔒 Booth版ビルド開始...")
            
            # 1. ディレクトリ準備
            self._prepare_directories()
            
            # 2. ファイル整合性チェック用ハッシュ生成
            file_hashes = self._generate_file_hashes()
            
            # 3. コード難読化
            print("🔀 コード難読化中...")
            self._obfuscate_source_code()
            
            # 4. 保護システム統合
            print("🛡️ 保護システム統合中...")
            self._integrate_protection_system(file_hashes)
            
            # 5. PyInstaller実行
            print("📦 バイナリ化中...")
            self._run_pyinstaller(entry_point, icon_path, additional_files)
            
            # 6. 追加保護処理
            print("🔐 追加保護処理中...")
            self._apply_post_build_protection()
            
            # 7. パッケージング
            print("📁 パッケージング中...")
            package_path = self._create_booth_package()
            
            print("✅ Booth版ビルド完了!")
            return {
                "success": True,
                "package_path": package_path,
                "protection_level": "MAXIMUM"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _prepare_directories(self):
        """ディレクトリ準備"""
        # クリーンアップ
        for dir_path in [self.build_dir, self.dist_dir, self.obfuscated_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_file_hashes(self) -> Dict[str, str]:
        """ファイルハッシュ生成"""
        file_hashes = {}
        python_files = list(self.project_root.glob("*.py"))
        
        for file_path in python_files:
            if file_path.name.startswith("booth_") or file_path.name.startswith("test_"):
                continue
                
            with open(file_path, 'rb') as f:
                content = f.read()
            
            file_hash = hashlib.sha256(content).hexdigest()
            file_hashes[str(file_path.relative_to(self.project_root))] = file_hash
        
        return file_hashes
    
    def _obfuscate_source_code(self):
        """ソースコード難読化"""
        python_files = list(self.project_root.glob("*.py"))
        
        for file_path in python_files:
            if file_path.name.startswith("booth_") or file_path.name.startswith("test_"):
                continue
            
            output_path = self.obfuscated_dir / file_path.name
            try:
                self.obfuscator.obfuscate_file(str(file_path), str(output_path))
            except Exception as e:
                print(f"難読化警告: {file_path.name} - {e}")
                # フォールバック: 元ファイルをコピー
                shutil.copy2(file_path, output_path)
    
    def _integrate_protection_system(self, file_hashes: Dict[str, str]):
        """保護システム統合"""
        # booth_protection.py を難読化版ディレクトリにコピー
        protection_file = self.project_root / "booth_protection.py"
        if protection_file.exists():
            shutil.copy2(protection_file, self.obfuscated_dir)
        
        # メインファイルに保護システムを統合
        main_files = ["main.py", "HAD_Statistics_GUI.py"]
        
        for main_file in main_files:
            main_path = self.obfuscated_dir / main_file
            if main_path.exists():
                self._inject_protection_code(main_path, file_hashes)
    
    def _inject_protection_code(self, main_file: Path, file_hashes: Dict[str, str]):
        """保護コード注入"""
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 保護システムインポートを追加
        protection_imports = """
# Booth Protection System
import sys
import os
from booth_protection import booth_protection, require_license, anti_debug

# 保護システム初期化
_protection_result = booth_protection.initialize_protection()
if not _protection_result["success"]:
    print("ライセンスエラー: " + _protection_result.get("error", "不明なエラー"))
    if "activation_url" in _protection_result.get("license_info", {}):
        print(f"アクティベーション: {_protection_result['license_info']['activation_url']}")
    sys.exit(1)
"""
        
        # ファイルの先頭（インポート後）に挿入
        lines = content.split('\n')
        insert_index = 0
        
        # インポート文の後を見つける
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_index = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        lines.insert(insert_index, protection_imports)
        
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _run_pyinstaller(self, entry_point: str, icon_path: str, additional_files: List[str]):
        """PyInstaller実行"""
        entry_file = self.obfuscated_dir / entry_point
        
        if not entry_file.exists():
            raise FileNotFoundError(f"エントリーポイント {entry_point} が見つかりません")
        
        # PyInstallerコマンド構築
        cmd = [
            "pyinstaller",
            "--onefile",  # 単一実行ファイル
            "--windowed",  # コンソールウィンドウを隠す
            "--clean",
            "--distpath", str(self.dist_dir),
            "--workpath", str(self.build_dir),
            "--specpath", str(self.build_dir),
            "--name", "StatisticsSuite_Booth",
        ]
        
        # アイコン指定
        if icon_path and Path(icon_path).exists():
            cmd.extend(["--icon", icon_path])
        
        # 追加ファイル
        if additional_files:
            for file_path in additional_files:
                if Path(file_path).exists():
                    cmd.extend(["--add-data", f"{file_path};."])
        
        # 隠しインポート（必要に応じて）
        hidden_imports = [
            "cryptography",
            "psutil",
            "requests",
            "PIL",
            "cv2",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "sklearn",
            "scipy",
            "tqdm"
        ]
        
        for imp in hidden_imports:
            cmd.extend(["--hidden-import", imp])
        
        # UPX圧縮（利用可能な場合）
        if shutil.which("upx"):
            cmd.append("--upx-dir=upx")
        
        cmd.append(str(entry_file))
        
        # 実行
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.obfuscated_dir)
        
        if result.returncode != 0:
            raise Exception(f"PyInstaller失敗: {result.stderr}")
    
    def _apply_post_build_protection(self):
        """ビルド後保護処理"""
        exe_files = list(self.dist_dir.glob("*.exe"))
        
        for exe_file in exe_files:
            try:
                # UPX圧縮（さらなる難読化）
                if shutil.which("upx"):
                    subprocess.run(["upx", "--best", "--lzma", str(exe_file)], 
                                 capture_output=True)
                
                # ファイル属性変更
                if os.name == 'nt':  # Windows
                    subprocess.run([
                        "attrib", "+R", "+H", str(exe_file)
                    ], capture_output=True)
                
            except Exception as e:
                print(f"後処理警告: {e}")
    
    def _create_booth_package(self) -> str:
        """Boothパッケージ作成"""
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