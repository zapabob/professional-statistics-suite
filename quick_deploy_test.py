#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡易版デプロイテストスクリプト
Professional Statistics Suite - Quick Deploy Test

Author: Professional Statistics Suite Team
Email: r.minegishi1987@gmail.com
License: MIT
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any

def test_basic_imports() -> Dict[str, Any]:
    """基本的なインポートテスト"""
    print("🔍 基本的なインポートテスト開始")
    
    results = {
        "numpy": False,
        "pandas": False,
        "matplotlib": False,
        "sklearn": False,
        "tkinter": False,
        "all_passed": False
    }
    
    try:
        import numpy as np
        results["numpy"] = True
        print("✅ NumPy インポート成功")
    except ImportError as e:
        print(f"❌ NumPy インポート失敗: {e}")
    
    try:
        import pandas as pd
        results["pandas"] = True
        print("✅ Pandas インポート成功")
    except ImportError as e:
        print(f"❌ Pandas インポート失敗: {e}")
    
    try:
        import matplotlib.pyplot as plt
        results["matplotlib"] = True
        print("✅ Matplotlib インポート成功")
    except ImportError as e:
        print(f"❌ Matplotlib インポート失敗: {e}")
    
    try:
        import sklearn
        results["sklearn"] = True
        print("✅ Scikit-learn インポート成功")
    except ImportError as e:
        print(f"❌ Scikit-learn インポート失敗: {e}")
    
    try:
        import tkinter as tk
        results["tkinter"] = True
        print("✅ Tkinter インポート成功")
    except ImportError as e:
        print(f"❌ Tkinter インポート失敗: {e}")
    
    results["all_passed"] = all([
        results["numpy"],
        results["pandas"],
        results["matplotlib"],
        results["sklearn"],
        results["tkinter"]
    ])
    
    return results

def test_data_processing() -> Dict[str, Any]:
    """データ処理テスト"""
    print("📊 データ処理テスト開始")
    
    try:
        import numpy as np
        import pandas as pd
        
        # テストデータ作成
        test_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice([1, 2, 3, 4, 5], 100)
        })
        
        # 基本的な統計計算
        mean_value = test_data['value'].mean()
        std_value = test_data['value'].std()
        
        print(f"✅ データ処理テスト成功: 平均={mean_value:.2f}, 標準偏差={std_value:.2f}")
        
        return {
            "success": True,
            "data_shape": test_data.shape,
            "mean_value": mean_value,
            "std_value": std_value
        }
        
    except Exception as e:
        print(f"❌ データ処理テスト失敗: {e}")
        return {"success": False, "error": str(e)}

def test_gui_components() -> Dict[str, Any]:
    """GUIコンポーネントテスト"""
    print("🖥️ GUIコンポーネントテスト開始")
    
    try:
        import tkinter as tk
        
        # 簡単なGUIテスト
        root = tk.Tk()
        root.withdraw()  # ウィンドウを非表示
        
        # ラベル作成
        label = tk.Label(root, text="テスト")
        
        # ボタン作成
        button = tk.Button(root, text="テストボタン")
        
        # エントリー作成
        entry = tk.Entry(root)
        
        root.destroy()
        
        print("✅ GUIコンポーネントテスト成功")
        
        return {
            "success": True,
            "components_tested": ["Label", "Button", "Entry"]
        }
        
    except Exception as e:
        print(f"❌ GUIコンポーネントテスト失敗: {e}")
        return {"success": False, "error": str(e)}

def test_file_operations() -> Dict[str, Any]:
    """ファイル操作テスト"""
    print("📁 ファイル操作テスト開始")
    
    try:
        # テストファイル作成
        test_file = "test_deploy.txt"
        test_content = "Professional Statistics Suite - デプロイテスト"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # ファイル読み込み
        with open(test_file, 'r', encoding='utf-8') as f:
            read_content = f.read()
        
        # ファイル削除
        os.remove(test_file)
        
        if read_content == test_content:
            print("✅ ファイル操作テスト成功")
            return {"success": True}
        else:
            print("❌ ファイル操作テスト失敗: 内容不一致")
            return {"success": False, "error": "内容不一致"}
        
    except Exception as e:
        print(f"❌ ファイル操作テスト失敗: {e}")
        return {"success": False, "error": str(e)}

def test_system_resources() -> Dict[str, Any]:
    """システムリソーステスト"""
    print("💻 システムリソーステスト開始")
    
    try:
        import psutil
        
        # メモリ使用量チェック
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        
        # CPU使用率チェック
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # ディスク使用量チェック
        disk = psutil.disk_usage('.')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        print(f"✅ システムリソーステスト成功:")
        print(f"   - メモリ使用量: {memory_usage_mb:.1f}MB")
        print(f"   - CPU使用率: {cpu_percent:.1f}%")
        print(f"   - ディスク使用率: {disk_usage_percent:.1f}%")
        
        return {
            "success": True,
            "memory_usage_mb": memory_usage_mb,
            "cpu_percent": cpu_percent,
            "disk_usage_percent": disk_usage_percent
        }
        
    except Exception as e:
        print(f"❌ システムリソーステスト失敗: {e}")
        return {"success": False, "error": str(e)}

def run_quick_deploy_test() -> Dict[str, Any]:
    """簡易版デプロイテスト実行"""
    print("簡易版デプロイテスト開始")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = time.time()
    
    # テスト実行
    test_results = {
        "imports": test_basic_imports(),
        "data_processing": test_data_processing(),
        "gui_components": test_gui_components(),
        "file_operations": test_file_operations(),
        "system_resources": test_system_resources()
    }
    
    # 結果集計
    successful_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if result.get("success", False) or result.get("all_passed", False):
            successful_tests += 1
    
    success_rate = (successful_tests / total_tests) * 100
    execution_time = time.time() - start_time
    
    # 結果表示
    print("\n" + "="*60)
    print("簡易版デプロイテスト結果")
    print("="*60)
    print(f"成功テスト数: {successful_tests}/{total_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"実行時間: {execution_time:.1f}秒")
    
    overall_success = success_rate >= 80.0
    
    if overall_success:
        print("デプロイテスト成功！本番環境デプロイ準備完了")
    else:
        print("デプロイテスト失敗。追加の修正が必要")
    
    return {
        "overall_success": overall_success,
        "success_rate": success_rate,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "execution_time": execution_time,
        "test_results": test_results
    }

def main():
    """メイン実行関数"""
    try:
        results = run_quick_deploy_test()
        
        # 結果をJSONファイルに保存
        with open("quick_deploy_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n結果保存: quick_deploy_test_results.json")
        
        if results["overall_success"]:
            print("\n次のステップ:")
            print("1. 本番環境デプロイ実行")
            print("2. パフォーマンス監視開始")
            print("3. ユーザーフィードバック収集")
        else:
            print("\n修正が必要な項目:")
            for test_name, result in results["test_results"].items():
                if not result.get("success", False) and not result.get("all_passed", False):
                    print(f"   - {test_name}: {result.get('error', '不明なエラー')}")
        
    except Exception as e:
        print(f"予期しないエラー: {e}")

if __name__ == "__main__":
    main() 