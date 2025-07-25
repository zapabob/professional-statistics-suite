#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOOTH License GUI Launcher
BOOTH ライセンス発行GUI システム起動ツール
"""

import sys
import os
import tkinter as tk
import subprocess

def check_dependencies():
    """依存関係チェック"""
    required_packages = [
        ('tkinter', 'Tkinter'),
        ('sqlite3', 'SQLite3'),
        ('qrcode', 'QRCode (pip install qrcode[pil])'),
        ('PIL', 'Pillow (pip install Pillow)')
    ]
    
    missing = []
    
    for package, display_name in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing.append(display_name)
    
    return missing

def install_missing_packages():
    """不足パッケージのインストール"""
    try:
        print("🔧 必要なパッケージをインストール中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qrcode[pil]", "Pillow"])
        print("✅ パッケージインストール完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ パッケージインストール失敗: {e}")
        return False

def main():
    """メイン実行"""
    print("🎯 BOOTH License Generator GUI Launcher")
    print("=" * 50)
    
    # 依存関係チェック
    missing = check_dependencies()
    
    if missing:
        print(f"⚠️  不足している依存関係: {', '.join(missing)}")
        
        # 自動インストール試行
        if 'QRCode' in str(missing) or 'Pillow' in str(missing):
            response = input("自動インストールを実行しますか？ (y/n): ")
            if response.lower() == 'y':
                if install_missing_packages():
                    print("✅ 依存関係の問題が解決されました")
                else:
                    print("❌ 自動インストールに失敗しました")
                    print("手動で以下のコマンドを実行してください:")
                    print("pip install qrcode[pil] Pillow")
                    return
            else:
                print("依存関係を手動でインストールしてから再実行してください")
                return
    
    # GUI起動
    try:
        print("🚀 BOOTH License Generator GUI を起動中...")
        
        # booth_license_gui.pyをインポートして実行
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from booth_license_gui import BoothLicenseGUI
        
        # Tkinter root作成
        root = tk.Tk()
        
        # アプリケーション起動
        app = BoothLicenseGUI(root)
        
        print("✅ GUI起動完了")
        print("📝 使用方法:")
        print("1. 商品番号 (7桁) を入力")
        print("2. 購入者固有番号 (8桁) を入力")
        print("3. 「ライセンス発行」ボタンをクリック")
        print("4. 生成されたライセンスキーを購入者に送付")
        print()
        print("ウィンドウを閉じるとプログラムが終了します。")
        
        # メインループ開始
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ GUI モジュールのインポートに失敗: {e}")
        print("booth_license_gui.py ファイルが存在することを確認してください")
    except Exception as e:
        print(f"❌ GUI起動に失敗: {e}")
        print("詳細なエラー情報:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 プログラムを終了しました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        input("Enterキーを押して終了...") 