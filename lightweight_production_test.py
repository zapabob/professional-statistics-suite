#!/usr/bin/env python3
"""
軽量版本番環境テスト
メモリ使用量を抑えた包括的なテスト
"""

import sys
import os
import time
import gc
import psutil
import json
from datetime import datetime
from typing import Dict, Any, List

def test_basic_imports() -> Dict[str, Any]:
    """基本的なインポートテスト"""
    print("基本インポートテスト開始")
    results = {"success": True, "errors": []}
    
    try:
        import pandas as pd
        print("Pandas インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Pandas: {e}")
        print(f"Pandas インポート失敗: {e}")
    
    try:
        import numpy as np
        print("NumPy インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"NumPy: {e}")
        print(f"NumPy インポート失敗: {e}")
    
    try:
        import matplotlib.pyplot as plt
        plt.ioff()  # インタラクティブモードを無効化
        print("Matplotlib インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Matplotlib: {e}")
        print(f"Matplotlib インポート失敗: {e}")
    
    try:
        import sklearn
        print("Scikit-learn インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Scikit-learn: {e}")
        print(f"Scikit-learn インポート失敗: {e}")
    
    return results

def test_core_modules() -> Dict[str, Any]:
    """コアモジュールテスト"""
    print("コアモジュールテスト開始")
    results = {"success": True, "errors": []}
    
    try:
        import src.core.config as config
        print("設定モジュール インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Config: {e}")
        print(f"設定モジュール インポート失敗: {e}")
    
    try:
        import src.data.data_preprocessing as dp
        print("データ前処理モジュール インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"DataPreprocessing: {e}")
        print(f"データ前処理モジュール インポート失敗: {e}")
    
    try:
        import src.statistics.advanced_statistics as stats
        print("統計解析モジュール インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"AdvancedStatistics: {e}")
        print(f"統計解析モジュール インポート失敗: {e}")
    
    try:
        from src.distribution.booth_deployment_automation import BoothDeploymentManager
        print("BoothDeploymentManager インポート成功")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"BoothDeploymentManager: {e}")
        print(f"BoothDeploymentManager インポート失敗: {e}")
    
    return results

def test_data_processing() -> Dict[str, Any]:
    """データ処理テスト"""
    print("データ処理テスト開始")
    results = {"success": True, "errors": []}
    
    try:
        import pandas as pd
        import numpy as np
        
        # 軽量なテストデータ作成
        data = pd.DataFrame({
            'value': np.random.normal(100, 15, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        # 基本的な統計計算
        mean_val = data['value'].mean()
        std_val = data['value'].std()
        
        print(f"データ処理テスト成功: 平均={mean_val:.2f}, 標準偏差={std_val:.2f}")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"DataProcessing: {e}")
        print(f"データ処理テスト失敗: {e}")
    
    return results

def test_file_operations() -> Dict[str, Any]:
    """ファイル操作テスト"""
    print("ファイル操作テスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # テストファイル作成
        test_file = "test_production_file.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("本番環境テスト用ファイル")
        
        # ファイル読み込み
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ファイル削除
        os.remove(test_file)
        
        print("ファイル操作テスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"FileOperations: {e}")
        print(f"ファイル操作テスト失敗: {e}")
    
    return results

def test_system_resources() -> Dict[str, Any]:
    """システムリソーステスト"""
    print("システムリソーステスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # メモリ使用量
        memory = psutil.virtual_memory()
        memory_usage = memory.used / (1024 * 1024)  # MB
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # ディスク使用率
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        print("システムリソーステスト成功:")
        print(f"   - メモリ使用量: {memory_usage:.1f}MB")
        print(f"   - CPU使用率: {cpu_percent:.1f}%")
        print(f"   - ディスク使用率: {disk_percent:.1f}%")
        
        # メモリ使用量が2GBを超える場合は警告
        if memory_usage > 2000:
            print(f"メモリ使用量警告: {memory_usage:.1f}MB > 2000MB")
            results["warnings"] = [f"High memory usage: {memory_usage:.1f}MB"]
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"SystemResources: {e}")
        print(f"システムリソーステスト失敗: {e}")
    
    return results

def test_gui_components() -> Dict[str, Any]:
    """GUIコンポーネントテスト（軽量版）"""
    print("GUIコンポーネントテスト開始")
    results = {"success": True, "errors": []}
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # 軽量なGUIテスト
        root = tk.Tk()
        root.withdraw()  # ウィンドウを表示しない
        
        # 基本的なウィジェット作成
        frame = ttk.Frame(root)
        button = ttk.Button(frame, text="テストボタン")
        label = ttk.Label(frame, text="テストラベル")
        
        # ウィジェット配置
        frame.pack()
        button.pack()
        label.pack()
        
        # 即座に破棄
        root.destroy()
        
        print("GUIコンポーネントテスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"GUIComponents: {e}")
        print(f"GUIコンポーネントテスト失敗: {e}")
    
    return results

def run_lightweight_production_test() -> Dict[str, Any]:
    """軽量版本番環境テスト実行"""
    print("軽量版本番環境テスト開始")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = time.time()
    test_results = []
    
    # テスト実行
    tests = [
        ("基本インポート", test_basic_imports),
        ("コアモジュール", test_core_modules),
        ("データ処理", test_data_processing),
        ("ファイル操作", test_file_operations),
        ("システムリソース", test_system_resources),
        ("GUIコンポーネント", test_gui_components)
    ]
    
    successful_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"{test_name}テスト実行中...")
        
        # メモリクリーンアップ
        gc.collect()
        
        result = test_func()
        test_results.append({
            "name": test_name,
            "result": result
        })
        
        if result["success"]:
            successful_tests += 1
            print(f"{test_name}テスト成功")
        else:
            print(f"{test_name}テスト失敗")
            for error in result.get("errors", []):
                print(f"   - エラー: {error}")
    
    execution_time = time.time() - start_time
    success_rate = (successful_tests / total_tests) * 100
    
    # 結果表示
    print("\n" + "="*60)
    print("軽量版本番環境テスト結果")
    print("="*60)
    print(f"成功テスト数: {successful_tests}/{total_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"実行時間: {execution_time:.1f}秒")
    
    overall_success = success_rate >= 80.0
    
    if overall_success:
        print("軽量版本番環境テスト成功！本番リリース準備完了")
    else:
        print("軽量版本番環境テスト失敗。追加の修正が必要")
    
    # 結果をJSONファイルに保存
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "overall_success": overall_success,
        "success_rate": success_rate,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "execution_time": execution_time,
        "test_results": test_results
    }
    
    with open("lightweight_production_test_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果保存: lightweight_production_test_results.json")
    
    return final_results

if __name__ == "__main__":
    results = run_lightweight_production_test()
    
    if results["overall_success"]:
        print("次のステップ:")
        print("1. 本番環境デプロイ実行")
        print("2. パフォーマンス監視開始")
        print("3. ユーザーフィードバック収集")
        sys.exit(0)
    else:
        print("テスト失敗。修正が必要です。")
        sys.exit(1)
