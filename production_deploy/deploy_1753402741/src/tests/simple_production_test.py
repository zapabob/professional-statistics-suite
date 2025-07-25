#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Production Environment Test
簡易本番環境テスト

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import os
import sys
import time
import psutil
import logging
import threading
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TestResult:
    """テスト結果"""
    test_name: str
    success: bool
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class SimpleProductionTest:
    """簡易本番環境テスト"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.results: List[TestResult] = []
        
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('production_test.log', encoding='utf-8')
            ]
        )
    
    def test_system_resources(self) -> TestResult:
        """システムリソーステスト"""
        print("🔍 システムリソーステスト実行中...")
        start_time = time.time()
        
        try:
            # CPU使用率を測定
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # メモリ使用量を測定
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            
            # ディスク使用量を確認
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            execution_time = time.time() - start_time
            
            # 閾値チェック
            success = (
                cpu_percent < 80 and 
                memory_mb < 2048 and  # 2GB以下
                disk_percent < 90
            )
            
            result = TestResult(
                test_name="System Resources Test",
                success=success,
                execution_time=execution_time,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            if not success:
                result.error_message = f"CPU: {cpu_percent}%, Memory: {memory_mb:.1f}MB, Disk: {disk_percent}%"
            
            print(f"✅ システムリソーステスト完了: CPU {cpu_percent}%, メモリ {memory_mb:.1f}MB")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name="System Resources Test",
                success=False,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e)
            )
            print(f"❌ システムリソーステスト失敗: {e}")
            return result
    
    def test_data_processing(self) -> TestResult:
        """データ処理テスト"""
        print("📊 データ処理テスト実行中...")
        start_time = time.time()
        
        try:
            # 大規模データセットを作成
            large_data = pd.DataFrame({
                'id': range(10000),
                'value': np.random.normal(100, 15, 10000),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
                'score': np.random.uniform(0, 100, 10000)
            })
            
            # データ処理操作
            processed_data = large_data.copy()
            processed_data['normalized_value'] = (processed_data['value'] - processed_data['value'].mean()) / processed_data['value'].std()
            processed_data['category_count'] = processed_data.groupby('category')['category'].transform('count')
            processed_data['score_rank'] = processed_data['score'].rank(ascending=False)
            
            # 統計計算
            stats = processed_data.describe()
            correlations = processed_data.corr()
            
            execution_time = time.time() - start_time
            
            # メモリ使用量を測定
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            result = TestResult(
                test_name="Data Processing Test",
                success=True,
                execution_time=execution_time,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            print(f"✅ データ処理テスト完了: {execution_time:.2f}秒, メモリ {memory_mb:.1f}MB")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name="Data Processing Test",
                success=False,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e)
            )
            print(f"❌ データ処理テスト失敗: {e}")
            return result
    
    def test_gui_creation(self) -> TestResult:
        """GUI作成テスト"""
        print("🎨 GUI作成テスト実行中...")
        start_time = time.time()
        
        try:
            # シンプルなTkinterウィンドウを作成
            root = tk.Tk()
            root.withdraw()  # ウィンドウを非表示
            
            # 基本的なGUI要素を作成
            frame = tk.Frame(root)
            frame.pack(padx=10, pady=10)
            
            label = tk.Label(frame, text="Production Test")
            label.pack()
            
            button = tk.Button(frame, text="Test Button")
            button.pack()
            
            entry = tk.Entry(frame)
            entry.pack()
            
            # GUIの応答性をテスト
            root.update()
            time.sleep(0.1)
            
            # ウィンドウを閉じる
            root.destroy()
            
            execution_time = time.time() - start_time
            
            # メモリ使用量を測定
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            result = TestResult(
                test_name="GUI Creation Test",
                success=True,
                execution_time=execution_time,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            print(f"✅ GUI作成テスト完了: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name="GUI Creation Test",
                success=False,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e)
            )
            print(f"❌ GUI作成テスト失敗: {e}")
            return result
    
    def test_file_operations(self) -> TestResult:
        """ファイル操作テスト"""
        print("📁 ファイル操作テスト実行中...")
        start_time = time.time()
        
        try:
            # テストディレクトリを作成
            test_dir = Path("test_production")
            test_dir.mkdir(exist_ok=True)
            
            # テストファイルを作成
            test_file = test_dir / "test_data.csv"
            test_data = pd.DataFrame({
                'id': range(1000),
                'value': np.random.randn(1000),
                'text': [f"test_{i}" for i in range(1000)]
            })
            
            # CSVファイルに保存
            test_data.to_csv(test_file, index=False)
            
            # ファイルを読み込み
            loaded_data = pd.read_csv(test_file)
            
            # JSONファイルに保存
            json_file = test_dir / "test_data.json"
            test_data.to_json(json_file, orient='records', indent=2)
            
            # ファイルサイズを確認
            file_size_mb = test_file.stat().st_size / 1024 / 1024
            
            # クリーンアップ
            test_file.unlink()
            json_file.unlink()
            test_dir.rmdir()
            
            execution_time = time.time() - start_time
            
            # メモリ使用量を測定
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            result = TestResult(
                test_name="File Operations Test",
                success=True,
                execution_time=execution_time,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            print(f"✅ ファイル操作テスト完了: {execution_time:.2f}秒, ファイルサイズ {file_size_mb:.2f}MB")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name="File Operations Test",
                success=False,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e)
            )
            print(f"❌ ファイル操作テスト失敗: {e}")
            return result
    
    def test_memory_stress(self) -> TestResult:
        """メモリストレステスト"""
        print("💾 メモリストレステスト実行中...")
        start_time = time.time()
        
        try:
            # 大量のデータを作成してメモリ使用量を増加
            data_structures = []
            
            for i in range(10):
                # 大きなDataFrameを作成
                large_df = pd.DataFrame({
                    'id': range(10000),
                    'value': np.random.randn(10000),
                    'category': np.random.choice(['A', 'B', 'C'], 10000)
                })
                data_structures.append(large_df)
                
                # メモリ使用量を確認
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if memory_mb > 1024:  # 1GBを超えたら停止
                    break
            
            # データ構造をクリア
            data_structures.clear()
            
            # ガベージコレクションを強制実行
            import gc
            gc.collect()
            
            execution_time = time.time() - start_time
            
            # 最終的なメモリ使用量を測定
            final_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            result = TestResult(
                test_name="Memory Stress Test",
                success=True,
                execution_time=execution_time,
                memory_usage_mb=final_memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            print(f"✅ メモリストレステスト完了: {execution_time:.2f}秒, 最終メモリ {final_memory_mb:.1f}MB")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name="Memory Stress Test",
                success=False,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e)
            )
            print(f"❌ メモリストレステスト失敗: {e}")
            return result
    
    def test_error_handling(self) -> TestResult:
        """エラーハンドリングテスト"""
        print("🛡️ エラーハンドリングテスト実行中...")
        start_time = time.time()
        
        try:
            # 意図的にエラーを発生させるテスト
            error_count = 0
            
            # 1. 存在しないファイルを読み込もうとする
            try:
                pd.read_csv("nonexistent_file.csv")
            except FileNotFoundError:
                error_count += 1
            
            # 2. 無効なデータ型で計算を試行
            try:
                invalid_data = pd.DataFrame({'text': ['a', 'b', 'c']})
                invalid_data['numeric'] = pd.to_numeric(invalid_data['text'], errors='coerce')
            except Exception:
                error_count += 1
            
            # 3. メモリ不足をシミュレート
            try:
                # 非常に大きな配列を作成しようとする
                large_array = np.zeros((10000, 10000))
            except MemoryError:
                error_count += 1
            except Exception:
                # MemoryError以外のエラーでもカウント
                error_count += 1
            
            execution_time = time.time() - start_time
            
            # メモリ使用量を測定
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # エラーが適切にハンドリングされたかチェック
            success = error_count >= 2  # 少なくとも2つのエラーがハンドリングされる
            
            result = TestResult(
                test_name="Error Handling Test",
                success=success,
                execution_time=execution_time,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            if not success:
                result.error_message = f"Only {error_count} errors were handled"
            
            print(f"✅ エラーハンドリングテスト完了: {error_count}個のエラーをハンドリング")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                test_name="Error Handling Test",
                success=False,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e)
            )
            print(f"❌ エラーハンドリングテスト失敗: {e}")
            return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """すべてのテストを実行"""
        print("🚀 本番環境テストスイート開始")
        print("=" * 50)
        
        # テスト関数のリスト
        test_functions = [
            self.test_system_resources,
            self.test_data_processing,
            self.test_gui_creation,
            self.test_file_operations,
            self.test_memory_stress,
            self.test_error_handling
        ]
        
        # 各テストを実行
        for test_func in test_functions:
            result = test_func()
            self.results.append(result)
            print()
        
        # 結果を集計
        summary = self.generate_summary()
        
        # 結果を保存
        self.save_results()
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """テスト結果のサマリーを生成"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        total_execution_time = sum(r.execution_time for r in self.results)
        total_memory_usage = sum(r.memory_usage_mb for r in self.results)
        average_cpu_usage = sum(r.cpu_usage_percent for r in self.results) / total_tests if total_tests > 0 else 0
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": total_execution_time,
                "average_execution_time": total_execution_time / total_tests if total_tests > 0 else 0,
                "total_memory_usage_mb": total_memory_usage,
                "average_memory_usage_mb": total_memory_usage / total_tests if total_tests > 0 else 0,
                "average_cpu_usage_percent": average_cpu_usage
            },
            "test_results": [r.__dict__ for r in self.results],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self):
        """テスト結果を保存"""
        summary = self.generate_summary()
        
        # JSONファイルに保存
        with open("production_test_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ テスト結果を保存: production_test_results.json")
    
    def print_summary(self):
        """サマリーを表示"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 50)
        print("📊 本番環境テスト結果サマリー")
        print("=" * 50)
        
        s = summary["summary"]
        print(f"✅ 総テスト数: {s['total_tests']}")
        print(f"✅ 成功テスト数: {s['successful_tests']}")
        print(f"❌ 失敗テスト数: {s['failed_tests']}")
        print(f"📈 成功率: {s['success_rate']:.1f}%")
        print(f"⏱️ 総実行時間: {s['total_execution_time']:.2f}秒")
        print(f"📊 平均実行時間: {s['average_execution_time']:.2f}秒")
        print(f"💾 平均メモリ使用量: {s['average_memory_usage_mb']:.1f}MB")
        print(f"🖥️ 平均CPU使用率: {s['average_cpu_usage_percent']:.1f}%")
        
        # 失敗したテストの詳細
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            print(f"\n❌ 失敗したテスト:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.error_message}")

def main():
    """メイン実行関数"""
    print("🚀 簡易本番環境テスト開始")
    
    # テスト実行
    tester = SimpleProductionTest()
    summary = tester.run_all_tests()
    
    # サマリー表示
    tester.print_summary()
    
    # 終了コード
    success_rate = summary["summary"]["success_rate"]
    if success_rate >= 80:
        print(f"\n🎉 テスト成功! 成功率: {success_rate:.1f}%")
        sys.exit(0)
    else:
        print(f"\n⚠️ テスト警告! 成功率: {success_rate:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    main() 