#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Statistics Suite GUI起動スクリプト
Professional Statistics Suite GUI Launcher

Author: Professional Statistics Suite Team
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import traceback

def check_dependencies():
    """依存関係のチェック"""
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'tkinter', 'json', 'threading', 'queue',
        'scipy', 'sklearn', 'statsmodels'
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
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                             'pandas', 'numpy', 'matplotlib', 'seaborn', 
                             'scipy', 'scikit-learn', 'statsmodels'])
        return True
    except subprocess.CalledProcessError:
        return False

def show_welcome_message():
    """ウェルカムメッセージの表示"""
    welcome_text = """
🚀 Professional Statistics Suite - Advanced Analytics 🚀

このアプリケーションは以下の高度な機能を提供します：

📊 データ管理
- CSVファイルの読み込み・保存
- 高度なデータ前処理
- 外れ値検出・処理

🤖 AI分析
- 自然言語クエリによる分析
- AIOrchestratorによる統合分析
- コンテキスト検索機能

📈 高度統計分析
- 記述統計、相関分析、回帰分析
- 分散分析、クラスター分析
- 因子分析、時系列分析、多変量分析

🔮 ベイズ分析
- ベイズ回帰・分類
- ベイズ検定・推定
- ベイズモデル比較

⏰ 生存時間分析
- Kaplan-Meier推定
- Cox比例ハザードモデル
- 生存関数・ハザード関数推定

⚡ 統計的検出力分析
- サンプルサイズ計算
- 検出力計算・効果量計算
- 検出力曲線プロット

📊 高度可視化
- ヒストグラム、散布図、箱ひげ図
- 相関行列、時系列プロット
- 密度プロット、QQプロット、残差プロット

🔍 仮定検証
- 正規性・等分散性検定
- 独立性・線形性検定
- 包括的仮定検証

📋 レポート生成
- 包括的レポート
- AI分析レポート
- 統計手法・ベイズ・生存時間分析レポート

🛡️ 監査・コンプライアンス
- 監査ログ表示
- コンプライアンスチェック
- データプライバシー・セキュリティ監査

📝 ログ管理
- 実装ログ表示
- システムログ管理

準備ができたら「OK」をクリックしてください。
"""
    
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを隠す
    
    result = messagebox.askokcancel("Professional Statistics Suite", welcome_text)
    root.destroy()
    
    return result

def main():
    """メイン関数"""
    print("🚀 Professional Statistics Suiteを起動中...")
    
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
    
    # Professional Statistics Suite GUIアプリの起動
    try:
        print("✅ Professional Statistics Suite GUIアプリを起動します...")
        
        # カレントディレクトリを設定
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # パスを追加
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'statistics'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gui'))
        
        # 高度なモジュールの事前インポートチェック
        try:
            import ai_integration
            print("✅ AI統合モジュール読み込み成功")
        except Exception as e:
            print(f"⚠️ AI統合モジュール読み込み警告: {e}")
        
        try:
            import statistical_method_advisor
            print("✅ 統計手法アドバイザーモジュール読み込み成功")
        except Exception as e:
            print(f"⚠️ 統計手法アドバイザーモジュール読み込み警告: {e}")
        
        try:
            import assumption_validator
            print("✅ 仮定検証モジュール読み込み成功")
        except Exception as e:
            print(f"⚠️ 仮定検証モジュール読み込み警告: {e}")
        
        # GUIアプリのインポートと起動
        from professional_statistics_gui import main as gui_main
        print("✅ GUIモジュール読み込み成功")
        gui_main()
        
    except ImportError as e:
        print(f"❌ モジュールのインポートエラー: {e}")
        print(f"詳細エラー情報: {traceback.format_exc()}")
        messagebox.showerror("エラー", f"モジュールのインポートエラー: {e}")
        
    except Exception as e:
        print(f"❌ アプリケーション起動エラー: {e}")
        print(f"詳細エラー情報: {traceback.format_exc()}")
        messagebox.showerror("エラー", f"アプリケーション起動エラー: {e}")

if __name__ == "__main__":
    main() 