#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automatic EXE Builder with License Protection
自動EXE化システム（ライセンス保護付き）
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies() -> bool:
    """依存関係チェック"""
    required_packages = ['pyinstaller', 'cryptography', 'psutil']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 不足パッケージ: {', '.join(missing)}")
        print("次のコマンドでインストールしてください:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def create_simple_protected_script(original_script: str, app_name: str) -> str:
    """簡易ライセンス保護スクリプト作成"""
    
    protected_code = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Protected {app_name}
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
import uuid
import platform

def get_machine_id():
    """マシンID取得"""
    try:
        mac = uuid.getnode()
        system = platform.system()
        machine = platform.machine()
        combined = f"{{mac}}{{system}}{{machine}}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    except:
        return "DEFAULT_MACHINE"

def check_trial():
    """14日間試用期間チェック"""
    machine_id = get_machine_id()
    trial_file = Path.home() / f".{{{app_name.lower().replace(' ', '_')}}}_trial.json"
    
    try:
        if trial_file.exists():
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)
            
            if trial_data.get('machine_id') != machine_id:
                print("❌ このマシンでは使用できません")
                return False
            
            start_date = datetime.fromisoformat(trial_data['start_date'])
            days_passed = (datetime.now() - start_date).days
            
            if days_passed >= 14:
                print("❌ 14日間の試用期間が終了しました")
                print("💰 ライセンスを購入してください")
                return False
            
            remaining = 14 - days_passed
            print(f"⏰ 試用期間: あと{{remaining}}日")
            return True
        else:
            # 初回起動
            trial_data = {{
                'machine_id': machine_id,
                'start_date': datetime.now().isoformat(),
                'app_name': '{app_name}'
            }}
            
            trial_file.parent.mkdir(exist_ok=True)
            with open(trial_file, 'w') as f:
                json.dump(trial_data, f)
            
            print("🎯 14日間の試用期間が開始されました")
            return True
            
    except Exception as e:
        print(f"❌ ライセンスチェックエラー: {{e}}")
        return False

def check_license():
    """ライセンスチェック"""
    license_file = Path("license.key")
    if license_file.exists():
        try:
            with open(license_file, 'r') as f:
                license_key = f.read().strip()
            
            # 簡単なライセンス検証
            if len(license_key) == 19 and license_key.count('-') == 3:
                print("✅ ライセンス認証済み")
                return True
        except:
            pass
    
    return check_trial()

def main():
    """メイン実行"""
    print(f"🚀 {app_name}")
    print("=" * 50)
    
    if not check_license():
        sys.exit(1)
    
    # オリジナルスクリプト実行
    try:
        # 現在のディレクトリにオリジナルスクリプトがあることを想定
        script_path = Path(__file__).parent / "{Path(original_script).name}"
        
        if script_path.exists():
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # メイン部分のみ実行（if __name__ == "__main__"以外）
            lines = script_content.split('\\n')
            main_code = []
            skip_main = False
            
            for line in lines:
                if 'if __name__ == "__main__"' in line:
                    skip_main = True
                    continue
                
                if not skip_main:
                    main_code.append(line)
            
            # 実行
            exec('\\n'.join(main_code), {{'__name__': '__main__'}})
        else:
            print(f"❌ スクリプトが見つかりません: {{script_path}}")
            
    except Exception as e:
        print(f"❌ エラー: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    return protected_code

def build_exe(script_path: str, app_name: str) -> bool:
    """EXE化実行"""
    try:
        print(f"🔨 {app_name} をEXE化中...")
        
        # 保護されたスクリプト作成
        protected_code = create_simple_protected_script(script_path, app_name)
        protected_file = f"protected_{Path(script_path).stem}.py"
        
        with open(protected_file, 'w', encoding='utf-8') as f:
            f.write(protected_code)
        
        # PyInstallerコマンド
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--clean",
            "--noconfirm",
            f"--name={Path(script_path).stem}",
            protected_file
        ]
        
        # GUIアプリの場合はコンソールを非表示
        if "gui" in script_path.lower() or "GUI" in script_path:
            cmd.append("--noconsole")
        
        print(f"📦 実行中: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 成功: {Path(script_path).stem}.exe")
            
            # 保護されたスクリプトを削除
            if os.path.exists(protected_file):
                os.remove(protected_file)
            
            return True
        else:
            print(f"❌ 失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🚀 Professional Statistics Suite - Auto EXE Builder")
    print("=" * 60)
    
    # 依存関係チェック
    if not check_dependencies():
        sys.exit(1)
    
    # EXE化対象スクリプト
    scripts = [
        ("main.py", "Professional Statistics Suite"),
        ("HAD_Statistics_GUI.py", "HAD Statistics GUI"), 
        ("booth_license_gui.py", "BOOTH License Manager"),
        ("booth_license_simple_gui.py", "BOOTH License Simple GUI"),
        ("run_web_dashboard.py", "Web Dashboard"),
        ("advanced_statistics.py", "Advanced Statistics"),
        ("advanced_visualization.py", "Data Visualization"),
        ("ai_integration.py", "AI Analysis Tool")
    ]
    
    success_count = 0
    total_count = 0
    
    for script_path, app_name in scripts:
        if Path(script_path).exists():
            total_count += 1
            if build_exe(script_path, app_name):
                success_count += 1
            print()
        else:
            print(f"⚠️  スクリプトが見つかりません: {script_path}")
    
    # 結果表示
    print("=" * 60)
    print(f"📊 結果: {success_count}/{total_count} 成功")
    
    if success_count > 0:
        print("📁 EXEファイル: ./dist/ フォルダに保存されました")
        print("⏰ 各EXEファイルには14日間の試用期間が設定されています")
        print("🔑 ライセンスキー (XXXX-XXXX-XXXX-XXXX形式) をlicense.keyファイルに保存することで制限が解除されます")
    
    # クリーンアップ
    for folder in ['build', '__pycache__']:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"🧹 {folder} フォルダを削除しました")

if __name__ == "__main__":
    main() 