# -*- coding: utf-8 -*-
"""
Kiro統合GUIアプリ起動スクリプト
Kiro Integrated GUI Application Launcher

Author: Kiro AI Assistant
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """依存関係のチェック"""
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'tkinter', 'json', 'threading', 'queue'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules

def install_dependencies():
    """依存関係のインストール"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy', 'matplotlib', 'seaborn'])
        return True
    except subprocess.CalledProcessError:
        return False

def show_welcome_message():
    """ウェルカムメッセージの表示"""
    welcome_text = """
🚀 Kiro統合統計分析システム 🚀

このアプリケーションは以下の機能を提供します：

📊 データ管理
- CSVファイルの読み込み・保存
- データ表示と情報表示

🤖 AI分析
- 自然言語クエリによる分析
- AIOrchestratorによる統合分析

📈 統計手法アドバイザー
- データ特性分析
- 統計手法推奨

🔍 仮説検証
- 正規性検定
- 等分散性検定
- 独立性検定

📊 可視化
- ヒストグラム、散布図、箱ひげ図、相関行列

📋 レポート生成
- 包括的レポート
- AI分析レポート
- 統計手法レポート

📝 ログ表示
- 実装ログの表示
- アプリケーションログ

準備ができたら「OK」をクリックしてください。
"""
    
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを隠す
    
    result = messagebox.askokcancel("Kiro統合統計分析システム", welcome_text)
    root.destroy()
    
    return result

def main():
    """メイン関数"""
    print("🚀 Kiro統合統計分析システムを起動中...")
    
    # 依存関係のチェック
    missing_modules = check_dependencies()
    
    if missing_modules:
        print(f"❌ 不足しているモジュール: {missing_modules}")
        
        root = tk.Tk()
        root.withdraw()
        
        install_choice = messagebox.askyesno(
            "依存関係エラー", 
            f"以下のモジュールが不足しています：\n{', '.join(missing_modules)}\n\nインストールしますか？"
        )
        
        if install_choice:
            if install_dependencies():
                messagebox.showinfo("成功", "依存関係のインストールが完了しました")
            else:
                messagebox.showerror("エラー", "依存関係のインストールに失敗しました")
                return
        else:
            return
        
        root.destroy()
    
    # ウェルカムメッセージの表示
    if not show_welcome_message():
        print("アプリケーションの起動をキャンセルしました")
        return
    
    # Kiro統合GUIアプリの起動
    try:
        print("✅ Kiro統合GUIアプリを起動します...")
        
        # カレントディレクトリを設定
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # GUIアプリのインポートと起動
        from kiro_integrated_gui import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"❌ モジュールのインポートエラー: {e}")
        messagebox.showerror("エラー", f"モジュールのインポートエラー: {e}")
        
    except Exception as e:
        print(f"❌ アプリケーション起動エラー: {e}")
        messagebox.showerror("エラー", f"アプリケーション起動エラー: {e}")

if __name__ == "__main__":
    main() 