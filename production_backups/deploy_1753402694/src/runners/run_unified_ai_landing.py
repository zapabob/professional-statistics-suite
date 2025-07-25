#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合AIランディングGUI起動スクリプト
Professional Statistics Suite - Unified AI Landing

Gemini、Claude、OpenAIの廉価版と最新モデルを統合したGUIランディングポイントを起動します。

Author: Professional Statistics Suite Team
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback
from datetime import datetime

def check_dependencies():
    """依存関係チェック"""
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'tkinter'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        error_msg = f"必要なモジュールが不足しています:\n{', '.join(missing_modules)}\n\n"
        error_msg += "以下のコマンドでインストールしてください:\n"
        error_msg += "pip install pandas numpy matplotlib seaborn"
        messagebox.showerror("依存関係エラー", error_msg)
        return False
    
    return True

def check_env_variables():
    """環境変数チェック"""
    required_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY", 
        "ANTHROPIC_API_KEY"
    ]
    
    optional_vars = [
        "TOGETHER_API_KEY",
        "OLLAMA_BASE_URL",
        "LMSTUDIO_BASE_URL",
        "KOBOLDCPP_BASE_URL"
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) in ["your_openai_api_key_here", "your_google_api_key_here", "your_anthropic_api_key_here"]:
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    return missing_required, missing_optional

def load_env_file():
    """環境変数ファイル読み込み"""
    env_file = ".env"
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
            print("✅ 環境変数ファイル(.env)を読み込みました")
            return True
        except Exception as e:
            print(f"❌ 環境変数ファイル読み込みエラー: {e}")
            return False
    else:
        print("⚠️ 環境変数ファイル(.env)が見つかりません")
        return False

def check_ai_modules():
    """AI統合モジュールチェック"""
    try:
        from ai_integration import AIOrchestrator, AnalysisContext
        print("✅ AI統合モジュール読み込み成功")
        return True
    except ImportError as e:
        print(f"❌ AI統合モジュール読み込み失敗: {e}")
        messagebox.showwarning("警告", "AI統合モジュールが利用できません。\n一部の機能が制限されます。")
        return False

def check_statistical_modules():
    """統計分析モジュールチェック"""
    try:
        from statistical_method_advisor import StatisticalMethodAdvisor
        print("✅ 統計手法アドバイザーモジュール読み込み成功")
        return True
    except ImportError as e:
        print(f"❌ 統計手法アドバイザーモジュール読み込み失敗: {e}")
        return False

def check_assumption_modules():
    """仮定検証モジュールチェック"""
    try:
        from assumption_validator import AssumptionValidator
        print("✅ 仮定検証モジュール読み込み成功")
        return True
    except ImportError as e:
        print(f"❌ 仮定検証モジュール読み込み失敗: {e}")
        return False

def check_gui_modules():
    """GUIモジュールチェック"""
    try:
        from unified_ai_landing_gui import UnifiedAILandingGUI
        print("✅ GUIモジュール読み込み成功")
        return True
    except ImportError as e:
        print(f"❌ GUIモジュール読み込み失敗: {e}")
        return False

def setup_environment():
    """環境設定"""
    # プロジェクトルートをパスに追加
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 必要なディレクトリを作成
    directories = [
        "unified_ai_backups",
        "checkpoints",
        "logs",
        "reports",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 環境変数ファイル読み込み
    load_env_file()
    
    print("✅ 環境設定完了")

def main():
    """メイン関数"""
    print("🚀 統合AIランディングGUIを起動中...")
    
    try:
        # 環境設定
        setup_environment()
        
        # 依存関係チェック
        if not check_dependencies():
            return
        
        # 環境変数チェック
        missing_required, missing_optional = check_env_variables()
        
        if missing_required:
            print("⚠️ 必要な環境変数が設定されていません:")
            for var in missing_required:
                print(f"  - {var}")
            print("\n環境変数ファイル(.env)を作成してAPIキーを設定してください。")
            print("env_template.txtを参考にしてください。")
        
        if missing_optional:
            print("⚠️ オプションの環境変数が設定されていません:")
            for var in missing_optional:
                print(f"  - {var}")
            print("ローカルLLMを使用する場合は設定してください。")
        
        # モジュールチェック
        ai_available = check_ai_modules()
        statistical_available = check_statistical_modules()
        assumption_available = check_assumption_modules()
        gui_available = check_gui_modules()
        
        if not gui_available:
            messagebox.showerror("エラー", "GUIモジュールが利用できません。\nunified_ai_landing_gui.pyファイルを確認してください。")
            return
        
        # GUI起動
        print("✅ 統合AIランディングGUIアプリを起動します...")
        
        # ルートウィンドウ作成
        root = tk.Tk()
        
        # アプリケーション作成
        from unified_ai_landing_gui import UnifiedAILandingGUI
        app = UnifiedAILandingGUI(root)
        
        # 終了処理
        def on_closing():
            try:
                print("Professional Statistics Suiteを終了します")
                root.destroy()
            except Exception as e:
                print(f"終了処理エラー: {e}")
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 起動完了メッセージ
        print("✅ 統合AIランディングGUI起動完了")
        print("📊 マルチプロバイダー対応AI統計分析システム")
        print("🤖 対応プロバイダー: Google Gemini, OpenAI, Anthropic Claude, ローカルLLM")
        
        if missing_required:
            print("⚠️ 一部の機能が制限されます（APIキー未設定）")
        
        # GUI実行
        root.mainloop()
        
    except Exception as e:
        error_msg = f"起動エラー: {e}\n\n詳細:\n{traceback.format_exc()}"
        print(error_msg)
        messagebox.showerror("起動エラー", error_msg)

if __name__ == "__main__":
    main() 