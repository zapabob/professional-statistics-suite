#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production Environment Test System
本番環境テストシステム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import time
import json
import psutil
import threading
import subprocess
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging
import traceback
import gc

# GUI関連
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np

# プロジェクト固有のインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# GUIモジュールのインポート（エラーハンドリング付き）
try:
    from src.gui.professional_statistics_gui import ProfessionalStatisticsGUI
    from src.gui.unified_ai_landing_gui import UnifiedAILandingGUI
    from src.gui.kiro_integrated_gui import KiroIntegratedGUI
    GUI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GUIモジュールのインポートエラー: {e}")
    print("📝 モックGUIクラスを使用します")
    GUI_MODULES_AVAILABLE = False
    
    # モックGUIクラス
    class MockGUI(tk.Frame):
        def __init__(self, root):
            super().__init__(root)
            self.root = root
            self.data = None
            
            # 基本的なウィジェットを作成
            self.create_basic_widgets()
        
        def create_basic_widgets(self):
            """基本的なウィジェットを作成"""
            # テスト用ボタンを作成
            self.test_button1 = tk.Button(self, text="テストボタン1", command=lambda: None)
            self.test_button1.pack(pady=5)
            
            self.test_button2 = tk.Button(self, text="テストボタン2", command=lambda: None)
            self.test_button2.pack(pady=5)
            
            self.test_button3 = tk.Button(self, text="テストボタン3", command=lambda: None)
            self.test_button3.pack(pady=5)
        
        def load_data(self, data):
            self.data = data
        
        def load_file(self, filepath):
            pass
        
        def fetch_online_data(self, url):
            pass
        
        def process_data(self, data):
            pass
        
        def perform_analysis(self):
            pass
    
    # 各GUIクラスのモック版
    class ProfessionalStatisticsGUI(MockGUI):
        def __init__(self, root):
            super().__init__(root)
    
    class UnifiedAILandingGUI(MockGUI):
        def __init__(self, root):
            super().__init__(root)
    
    class KiroIntegratedGUI(MockGUI):
        def __init__(self, root):
            super().__init__(root)
# from gui.HAD_Statistics_GUI import HADStatisticsGUI  # モジュール不足のため一時的に無効化

class ProductionTestResult:
    """本番環境テスト結果クラス"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = datetime.now()
        self.end_time = None
        self.success = False
        self.error_message = None
        self.performance_metrics = {}
        self.memory_usage = {}
        self.cpu_usage = {}
        self.gui_responsiveness = {}
        self.data_processing_time = {}
    
    def complete(self, success: bool, error_message: str = None):
        """テスト完了"""
        self.end_time = datetime.now()
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict:
        """辞書形式で結果を返す"""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "success": self.success,
            "error_message": self.error_message,
            "performance_metrics": self.performance_metrics,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "gui_responsiveness": self.gui_responsiveness,
            "data_processing_time": self.data_processing_time
        }

class ProductionEnvironmentTest:
    """本番環境テストシステム"""
    
    def __init__(self):
        self.test_results = {}
        self.gui_instances = {}
        self.monitoring_active = False
        self.performance_monitor = PerformanceMonitor()
        
        # ログ設定
        self._setup_logging()
        
        # テストデータ
        self.test_data = self._create_production_test_data()
        
        # テスト設定
        self.test_config = {
            "memory_threshold_mb": 1024,  # 1GB
            "cpu_threshold_percent": 80,  # 80%
            "response_time_threshold_ms": 5000,  # 5秒
            "max_test_duration_minutes": 30,
            "data_size_threshold_mb": 100  # 100MB
        }
    
    def _setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"production_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_production_test_data(self) -> Dict[str, pd.DataFrame]:
        """本番環境用テストデータ作成（改善版）"""
        try:
            # 数値データのみのデータセット（文字列データを排除）
            numerical_dataset = pd.DataFrame({
                'id': range(10000),
                'value': np.random.normal(100, 15, 10000),
                'score': np.random.uniform(0, 100, 10000),
                'count': np.random.poisson(50, 10000),
                'ratio': np.random.beta(2, 5, 10000),
                'timestamp': pd.date_range('2023-01-01', periods=10000, freq='H')
            })
            
            # 複合データセット（カテゴリカルデータは別途処理）
            complex_dataset = pd.DataFrame({
                'x1': np.random.normal(0, 1, 5000),
                'x2': np.random.normal(0, 1, 5000),
                'x3': np.random.normal(0, 1, 5000),
                'y': np.random.normal(0, 1, 5000),
                'weight': np.random.exponential(1, 5000),
                'probability': np.random.beta(1, 1, 5000)
            })
            
            # 時系列データセット
            timeseries_large = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5000, freq='15min'),
                'value': np.random.normal(100, 10, 5000).cumsum(),
                'volume': np.random.poisson(1000, 5000),
                'price': np.random.lognormal(4, 0.5, 5000),
                'volatility': np.random.gamma(2, 0.5, 5000)
            })
            
            # データ型検証
            self._validate_test_data(numerical_dataset, "numerical_dataset")
            self._validate_test_data(complex_dataset, "complex_dataset")
            self._validate_test_data(timeseries_large, "timeseries_large")
            
            return {
                "numerical_dataset": numerical_dataset,
                "complex_dataset": complex_dataset,
                "timeseries_large": timeseries_large
            }
            
        except Exception as e:
            self.logger.error(f"❌ テストデータ生成エラー: {e}")
            # フォールバック用の最小データセット
            return {
                "fallback_dataset": pd.DataFrame({
                    'id': range(100),
                    'value': np.random.normal(0, 1, 100)
                })
            }
    
    def _validate_test_data(self, df: pd.DataFrame, dataset_name: str):
        """テストデータの検証"""
        try:
            # 数値列の検証
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    self.logger.warning(f"⚠️ {dataset_name}.{col} に欠損値が含まれています")
                
                if np.isinf(df[col]).any():
                    self.logger.warning(f"⚠️ {dataset_name}.{col} に無限値が含まれています")
            
            # データサイズの検証
            if df.empty:
                raise ValueError(f"{dataset_name} が空のデータフレームです")
            
            self.logger.info(f"✅ {dataset_name} データ検証完了: {df.shape[0]}行 x {df.shape[1]}列")
            
        except Exception as e:
            self.logger.error(f"❌ {dataset_name} データ検証エラー: {e}")
            raise
    
    def test_gui_startup_performance(self, gui_class, test_name: str) -> ProductionTestResult:
        """GUI起動パフォーマンステスト"""
        self.logger.info(f"🧪 GUI起動パフォーマンステスト開始: {test_name}")
        
        result = ProductionTestResult(f"{test_name}_startup")
        
        try:
            # メモリ使用量記録開始
            self.performance_monitor.start_monitoring()
            
            # GUI起動時間測定
            start_time = time.time()
            
            root = tk.Tk()
            root.withdraw()
            
            # メインループを開始（非ブロッキング）
            root.after(100, lambda: None)  # イベントループを初期化
            
            gui_instance = gui_class(root)
            self.gui_instances[test_name] = gui_instance
            
            startup_time = time.time() - start_time
            
            # パフォーマンスメトリクス記録
            result.performance_metrics = {
                "startup_time_seconds": startup_time,
                "memory_usage_mb": self.performance_monitor.get_memory_usage_mb(),
                "cpu_usage_percent": self.performance_monitor.get_cpu_usage_percent()
            }
            
            # 起動時間チェック
            if startup_time > 10:  # 10秒以上で警告
                self.logger.warning(f"⚠️ {test_name} 起動時間が長い: {startup_time:.2f}秒")
            
            # メモリ使用量チェック
            memory_usage = result.performance_metrics["memory_usage_mb"]
            if memory_usage > self.test_config["memory_threshold_mb"]:
                self.logger.warning(f"⚠️ {test_name} メモリ使用量が高い: {memory_usage:.1f}MB")
            
            result.complete(True)
            self.logger.info(f"✅ {test_name} 起動パフォーマンステスト完了: {startup_time:.2f}秒")
            
            return result
            
        except Exception as e:
            result.complete(False, str(e))
            self.logger.error(f"❌ {test_name} 起動パフォーマンステスト失敗: {e}")
            return result
    
    def test_large_data_processing(self, gui_class, test_name: str) -> ProductionTestResult:
        """大規模データ処理テスト"""
        self.logger.info(f"🧪 大規模データ処理テスト開始: {test_name}")
        
        result = ProductionTestResult(f"{test_name}_large_data")
        
        try:
            # GUIインスタンス作成
            root = tk.Tk()
            root.withdraw()
            
            gui_instance = gui_class(root)
            self.gui_instances[test_name] = gui_instance
            
            # 大規模データセットでテスト
            large_data = self.test_data["numerical_dataset"] # 改善版ではnumerical_datasetを使用
            
            # データ読み込み時間測定
            start_time = time.time()
            
            if hasattr(gui_instance, 'load_data'):
                gui_instance.load_data(large_data)
            
            load_time = time.time() - start_time
            
            # 統計分析実行時間測定
            analysis_start = time.time()
            
            if hasattr(gui_instance, 'perform_analysis'):
                gui_instance.perform_analysis()
            
            analysis_time = time.time() - analysis_start
            
            # パフォーマンスメトリクス記録
            result.performance_metrics = {
                "data_size_mb": large_data.memory_usage(deep=True).sum() / 1024 / 1024,
                "load_time_seconds": load_time,
                "analysis_time_seconds": analysis_time,
                "total_processing_time_seconds": load_time + analysis_time,
                "memory_usage_mb": self.performance_monitor.get_memory_usage_mb(),
                "cpu_usage_percent": self.performance_monitor.get_cpu_usage_percent()
            }
            
            # 処理時間チェック
            total_time = result.performance_metrics["total_processing_time_seconds"]
            if total_time > 30:  # 30秒以上で警告
                self.logger.warning(f"⚠️ {test_name} データ処理時間が長い: {total_time:.2f}秒")
            
            result.complete(True)
            self.logger.info(f"✅ {test_name} 大規模データ処理テスト完了: {total_time:.2f}秒")
            
            # クリーンアップ
            root.destroy()
            
            return result
            
        except Exception as e:
            result.complete(False, str(e))
            self.logger.error(f"❌ {test_name} 大規模データ処理テスト失敗: {e}")
            return result
    
    def test_gui_responsiveness(self, gui_class, test_name: str) -> ProductionTestResult:
        """GUI応答性テスト（改善版）"""
        self.logger.info(f"🧪 GUI応答性テスト開始: {test_name}")
        
        result = ProductionTestResult(f"{test_name}_responsiveness")
        
        try:
            # GUIインスタンス作成
            root = tk.Tk()
            root.withdraw()
            
            # メインループを開始（非ブロッキング）
            root.after(100, lambda: None)  # イベントループを初期化
            
            gui_instance = gui_class(root)
            self.gui_instances[test_name] = gui_instance
            
            # ボタン応答時間テスト（改善版）
            buttons = self._find_all_buttons_improved(gui_instance)
            response_times = []
            button_test_results = []
            
            self.logger.info(f"🔍 検出されたボタン数: {len(buttons)}")
            
            for button_name, button_widget in list(buttons.items())[:5]:  # 最初の5つのボタンのみテスト
                try:
                    if self._is_button_enabled_improved(button_widget):
                        # ボタンクリック応答時間測定
                        start_time = time.time()
                        
                        # ボタンクリック実行
                        button_widget.invoke()
                        
                        response_time = (time.time() - start_time) * 1000  # ミリ秒
                        response_times.append(response_time)
                        
                        # テスト結果記録
                        button_result = {
                            "button_name": button_name,
                            "response_time_ms": response_time,
                            "success": response_time < self.test_config["response_time_threshold_ms"]
                        }
                        button_test_results.append(button_result)
                        
                        # 応答時間チェック
                        if response_time > self.test_config["response_time_threshold_ms"]:
                            self.logger.warning(f"⚠️ {button_name} 応答時間が長い: {response_time:.1f}ms")
                        else:
                            self.logger.info(f"✅ {button_name} 応答時間良好: {response_time:.1f}ms")
                        
                        time.sleep(0.1)  # 100ms待機
                    else:
                        self.logger.info(f"⏭️ {button_name} は無効状態のためスキップ")
                        
                except Exception as e:
                    self.logger.error(f"❌ ボタンテストエラー {button_name}: {e}")
                    button_result = {
                        "button_name": button_name,
                        "response_time_ms": 0,
                        "success": False,
                        "error": str(e)
                    }
                    button_test_results.append(button_result)
            
            # 応答性メトリクス記録
            result.gui_responsiveness = {
                "average_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "buttons_tested": len(response_times),
                "response_times": response_times,
                "button_test_results": button_test_results,
                "total_buttons_found": len(buttons)
            }
            
            # 成功率計算
            successful_tests = sum(1 for r in button_test_results if r.get("success", False))
            total_tests = len(button_test_results)
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            result.complete(True)
            self.logger.info(f"✅ {test_name} GUI応答性テスト完了: 平均応答時間 {result.gui_responsiveness['average_response_time_ms']:.1f}ms, 成功率 {success_rate:.1f}%")
            
            # クリーンアップ
            root.destroy()
            
            return result
            
        except Exception as e:
            result.complete(False, str(e))
            self.logger.error(f"❌ {test_name} GUI応答性テスト失敗: {e}")
            return result
    
    def test_memory_leak_detection(self, gui_class, test_name: str) -> ProductionTestResult:
        """メモリリーク検出テスト"""
        self.logger.info(f"🧪 メモリリーク検出テスト開始: {test_name}")
        
        result = ProductionTestResult(f"{test_name}_memory_leak")
        
        try:
            # ガベージコレクション実行
            gc.collect()
            
            # 初期メモリ使用量記録
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            memory_usage_history = [initial_memory]
            
            # 複数回のGUI操作でメモリ使用量を監視
            for i in range(10):
                root = tk.Tk()
                root.withdraw()
                
                # メインループを開始（非ブロッキング）
                root.after(100, lambda: None)  # イベントループを初期化
                
                gui_instance = gui_class(root)
                
                # テストデータ読み込み
                test_data = self.test_data["complex_dataset"] # complex_datasetを使用
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(test_data)
                
                # 分析実行
                if hasattr(gui_instance, 'perform_analysis'):
                    gui_instance.perform_analysis()
                
                # メモリ使用量記録
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage_history.append(current_memory)
                
                # クリーンアップ
                root.destroy()
                gc.collect()
                
                time.sleep(0.5)  # 500ms待機
            
            # メモリリーク分析
            memory_increase = memory_usage_history[-1] - memory_usage_history[0]
            memory_growth_rate = memory_increase / len(memory_usage_history)
            
            result.memory_usage = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": memory_usage_history[-1],
                "memory_increase_mb": memory_increase,
                "memory_growth_rate_mb_per_cycle": memory_growth_rate,
                "memory_usage_history": memory_usage_history,
                "potential_memory_leak": memory_increase > 50  # 50MB以上の増加でリークと判定
            }
            
            # メモリリークチェック
            if result.memory_usage["potential_memory_leak"]:
                self.logger.warning(f"⚠️ {test_name} メモリリークの可能性: {memory_increase:.1f}MB増加")
            
            result.complete(True)
            self.logger.info(f"✅ {test_name} メモリリーク検出テスト完了: メモリ増加 {memory_increase:.1f}MB")
            
            return result
            
        except Exception as e:
            result.complete(False, str(e))
            self.logger.error(f"❌ {test_name} メモリリーク検出テスト失敗: {e}")
            return result
    
    def test_error_handling_and_recovery(self, gui_class, test_name: str) -> ProductionTestResult:
        """エラーハンドリングとリカバリーテスト"""
        self.logger.info(f"🧪 エラーハンドリングとリカバリーテスト開始: {test_name}")
        
        result = ProductionTestResult(f"{test_name}_error_handling")
        
        try:
            # GUIインスタンス作成
            root = tk.Tk()
            root.withdraw()
            
            # メインループを開始（非ブロッキング）
            root.after(100, lambda: None)  # イベントループを初期化
            
            gui_instance = gui_class(root)
            self.gui_instances[test_name] = gui_instance
            
            error_tests = [
                self._test_invalid_data_handling,
                self._test_missing_file_handling,
                self._test_network_error_handling,
                self._test_memory_error_handling
            ]
            
            error_results = []
            
            for error_test in error_tests:
                try:
                    error_result = error_test(gui_instance)
                    error_results.append(error_result)
                except Exception as e:
                    error_results.append({
                        "test_type": error_test.__name__,
                        "success": False,
                        "error": str(e)
                    })
            
            # エラーハンドリング結果記録
            result.performance_metrics = {
                "total_error_tests": len(error_tests),
                "successful_error_handling": sum(1 for r in error_results if r.get("success", False)),
                "failed_error_handling": sum(1 for r in error_results if not r.get("success", False)),
                "error_handling_rate": sum(1 for r in error_results if r.get("success", False)) / len(error_tests) * 100
            }
            
            result.complete(True)
            self.logger.info(f"✅ {test_name} エラーハンドリングテスト完了: {result.performance_metrics['successful_error_handling']}/{result.performance_metrics['total_error_tests']} 成功")
            
            # クリーンアップ
            root.destroy()
            
            return result
            
        except Exception as e:
            result.complete(False, str(e))
            self.logger.error(f"❌ {test_name} エラーハンドリングテスト失敗: {e}")
            return result
    
    def _test_invalid_data_handling(self, gui_instance) -> Dict:
        """無効データハンドリングテスト（改善版）"""
        error_count = 0
        total_tests = 0
        
        try:
            # テスト1: 文字列データを含むデータフレーム
            total_tests += 1
            try:
                invalid_data1 = pd.DataFrame({
                    'numeric': [1, 2, 3],
                    'string': ['A', 'B', 'C'],
                    'mixed': [1, 'string', 3.14]
                })
                
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(invalid_data1)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（文字列データ）: {e}")
            
            # テスト2: 欠損値のみのデータフレーム
            total_tests += 1
            try:
                invalid_data2 = pd.DataFrame({
                    'null_col': [None, None, None],
                    'empty_col': ['', '', '']
                })
                
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(invalid_data2)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（欠損値）: {e}")
            
            # テスト3: 空のデータフレーム
            total_tests += 1
            try:
                empty_data = pd.DataFrame()
                
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(empty_data)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（空データ）: {e}")
            
            # テスト4: 無限値を含むデータフレーム
            total_tests += 1
            try:
                infinite_data = pd.DataFrame({
                    'normal': [1, 2, 3],
                    'inf': [1, np.inf, 3],
                    'nan': [1, np.nan, 3]
                })
                
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(infinite_data)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（無限値）: {e}")
            
            success_rate = (total_tests - error_count) / total_tests * 100
            return {
                "test_type": "invalid_data", 
                "success": success_rate >= 50,  # 50%以上のエラーハンドリングで成功
                "success_rate": success_rate,
                "total_tests": total_tests,
                "handled_errors": total_tests - error_count
            }
            
        except Exception as e:
            return {"test_type": "invalid_data", "success": False, "error": str(e)}
    
    def _test_missing_file_handling(self, gui_instance) -> Dict:
        """ファイル不存在ハンドリングテスト（改善版）"""
        error_count = 0
        total_tests = 0
        
        try:
            # テスト1: 存在しないCSVファイル
            total_tests += 1
            try:
                non_existent_csv = "/path/to/non/existent/file.csv"
                
                if hasattr(gui_instance, 'load_file'):
                    gui_instance.load_file(non_existent_csv)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（存在しないCSV）: {e}")
            
            # テスト2: 存在しないExcelファイル
            total_tests += 1
            try:
                non_existent_excel = "/path/to/non/existent/file.xlsx"
                
                if hasattr(gui_instance, 'load_file'):
                    gui_instance.load_file(non_existent_excel)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（存在しないExcel）: {e}")
            
            # テスト3: 権限のないファイル
            total_tests += 1
            try:
                no_permission_file = "/root/system_file.txt"
                
                if hasattr(gui_instance, 'load_file'):
                    gui_instance.load_file(no_permission_file)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（権限なし）: {e}")
            
            success_rate = (total_tests - error_count) / total_tests * 100
            return {
                "test_type": "missing_file", 
                "success": success_rate >= 50,
                "success_rate": success_rate,
                "total_tests": total_tests,
                "handled_errors": total_tests - error_count
            }
            
        except Exception as e:
            return {"test_type": "missing_file", "success": False, "error": str(e)}
    
    def _test_network_error_handling(self, gui_instance) -> Dict:
        """ネットワークエラーハンドリングテスト（改善版）"""
        error_count = 0
        total_tests = 0
        
        try:
            # テスト1: 無効なURL
            total_tests += 1
            try:
                invalid_url = "http://invalid-url-that-will-fail.com"
                
                if hasattr(gui_instance, 'fetch_online_data'):
                    gui_instance.fetch_online_data(invalid_url)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（無効URL）: {e}")
            
            # テスト2: タイムアウトをシミュレート
            total_tests += 1
            try:
                timeout_url = "http://httpbin.org/delay/10"  # 10秒遅延
                
                if hasattr(gui_instance, 'fetch_online_data'):
                    gui_instance.fetch_online_data(timeout_url)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（タイムアウト）: {e}")
            
            # テスト3: 404エラー
            total_tests += 1
            try:
                not_found_url = "http://httpbin.org/status/404"
                
                if hasattr(gui_instance, 'fetch_online_data'):
                    gui_instance.fetch_online_data(not_found_url)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（404）: {e}")
            
            success_rate = (total_tests - error_count) / total_tests * 100
            return {
                "test_type": "network_error", 
                "success": success_rate >= 50,
                "success_rate": success_rate,
                "total_tests": total_tests,
                "handled_errors": total_tests - error_count
            }
            
        except Exception as e:
            return {"test_type": "network_error", "success": False, "error": str(e)}
    
    def _test_memory_error_handling(self, gui_instance) -> Dict:
        """メモリエラーハンドリングテスト（改善版）"""
        error_count = 0
        total_tests = 0
        
        try:
            # テスト1: 非常に大きなデータセット
            total_tests += 1
            try:
                large_data = pd.DataFrame({
                    'data': np.random.random(1000000)  # 1M行（メモリ使用量を制限）
                })
                
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(large_data)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（大容量データ）: {e}")
            
            # テスト2: メモリリークをシミュレート
            total_tests += 1
            try:
                # 大量のオブジェクトを作成
                objects = []
                for i in range(100000):
                    objects.append(f"object_{i}" * 100)
                
                if hasattr(gui_instance, 'process_data'):
                    gui_instance.process_data(objects)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（メモリリーク）: {e}")
            
            # テスト3: 無効なメモリ操作
            total_tests += 1
            try:
                # 無効なメモリアドレスをシミュレート
                invalid_data = None
                
                if hasattr(gui_instance, 'load_data'):
                    gui_instance.load_data(invalid_data)
            except Exception as e:
                error_count += 1
                self.logger.debug(f"期待されるエラー（無効メモリ）: {e}")
            
            success_rate = (total_tests - error_count) / total_tests * 100
            return {
                "test_type": "memory_error", 
                "success": success_rate >= 50,
                "success_rate": success_rate,
                "total_tests": total_tests,
                "handled_errors": total_tests - error_count
            }
            
        except Exception as e:
            return {"test_type": "memory_error", "success": False, "error": str(e)}
    
    def _find_all_buttons(self, widget) -> Dict[str, tk.Widget]:
        """ウィジェット内の全てのボタンを再帰的に検索"""
        buttons = {}
        
        def search_buttons(w):
            if isinstance(w, (tk.Button, ttk.Button)):
                button_name = self._get_button_name(w)
                buttons[button_name] = w
            
            # 子ウィジェットを再帰的に検索
            for child in w.winfo_children():
                search_buttons(child)
        
        search_buttons(widget)
        return buttons
    
    def _get_button_name(self, button_widget: tk.Widget) -> str:
        """ボタン名を取得"""
        try:
            if hasattr(button_widget, 'cget'):
                text = button_widget.cget("text")
                if text:
                    return text.strip()
            return f"Button_{id(button_widget)}"
        except Exception:
            return f"Button_{id(button_widget)}"
    
    def _is_button_enabled(self, button_widget: tk.Widget) -> bool:
        """ボタンが有効かチェック"""
        try:
            if hasattr(button_widget, 'cget'):
                state = button_widget.cget("state")
                return state != "disabled"
            return True
        except Exception:
            return True
    
    def _find_all_buttons_improved(self, widget) -> Dict[str, tk.Widget]:
        """ウィジェット内の全てのボタンを再帰的に検索（改善版）"""
        buttons = {}
        
        def search_buttons(w):
            try:
                # ボタンウィジェットの検出
                if isinstance(w, (tk.Button, ttk.Button)):
                    button_name = self._get_button_name(w)
                    buttons[button_name] = w
                
                # 子ウィジェットを再帰的に検索
                if hasattr(w, 'winfo_children'):
                    for child in w.winfo_children():
                        search_buttons(child)
                        
            except Exception as e:
                self.logger.debug(f"ウィジェット検索エラー: {e}")
        
        try:
            search_buttons(widget)
        except Exception as e:
            self.logger.error(f"ボタン検索エラー: {e}")
        
        return buttons
    
    def _is_button_enabled_improved(self, button_widget: tk.Widget) -> bool:
        """ボタンが有効かチェック（改善版）"""
        try:
            if hasattr(button_widget, 'cget'):
                state = button_widget.cget("state")
                return state != "disabled"
            return True
        except Exception as e:
            self.logger.debug(f"ボタン状態チェックエラー: {e}")
            return True
    
    def run_comprehensive_production_test(self) -> Dict:
        """包括的な本番環境テスト実行（改善版）"""
        self.logger.info("🚀 包括的な本番環境テスト開始")
        
        start_time = time.time()
        
        try:
            # テスト対象のGUIクラス（モック対応）
            gui_classes = []
            
            if GUI_MODULES_AVAILABLE:
                gui_classes = [
                    (ProfessionalStatisticsGUI, "ProfessionalStatisticsGUI"),
                    (UnifiedAILandingGUI, "UnifiedAILandingGUI"),
                    (KiroIntegratedGUI, "KiroIntegratedGUI")
                ]
            else:
                # モックGUIクラスを使用
                gui_classes = [
                    (ProfessionalStatisticsGUI, "MockProfessionalStatisticsGUI"),
                    (UnifiedAILandingGUI, "MockUnifiedAILandingGUI"),
                    (KiroIntegratedGUI, "MockKiroIntegratedGUI")
                ]
                self.logger.info("📝 モックGUIクラスを使用してテストを実行します")
            
            # 各GUIの本番環境テスト
            for gui_class, gui_name in gui_classes:
                self.logger.info(f"🧪 {gui_name} 本番環境テスト開始")
                
                try:
                    # 起動パフォーマンステスト
                    startup_result = self.test_gui_startup_performance(gui_class, gui_name)
                    self.test_results[f"{gui_name}_startup"] = startup_result
                    
                    # 大規模データ処理テスト
                    data_result = self.test_large_data_processing(gui_class, gui_name)
                    self.test_results[f"{gui_name}_data"] = data_result
                    
                    # GUI応答性テスト
                    response_result = self.test_gui_responsiveness(gui_class, gui_name)
                    self.test_results[f"{gui_name}_response"] = response_result
                    
                    # メモリリーク検出テスト
                    memory_result = self.test_memory_leak_detection(gui_class, gui_name)
                    self.test_results[f"{gui_name}_memory"] = memory_result
                    
                    # エラーハンドリングテスト
                    error_result = self.test_error_handling_and_recovery(gui_class, gui_name)
                    self.test_results[f"{gui_name}_error"] = error_result
                    
                    self.logger.info(f"✅ {gui_name} テスト完了")
                    
                except Exception as e:
                    self.logger.error(f"❌ {gui_name} テスト失敗: {e}")
                    # エラーが発生しても他のテストは続行
                    continue
            
            # 結果集計
            total_tests = len(self.test_results)
            successful_tests = sum(1 for result in self.test_results.values() if result.success)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # パフォーマンス監視の停止
            self.performance_monitor.stop_monitoring()
            
            final_results = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "duration_seconds": duration,
                "test_details": {name: result.to_dict() for name, result in self.test_results.items()},
                "performance_summary": self._generate_performance_summary(),
                "performance_monitor_summary": self.performance_monitor.get_performance_summary()
            }
            
            # 結果保存
            self._save_test_results(final_results)
            
            self.logger.info(f"✅ 包括的な本番環境テスト完了: {successful_tests}/{total_tests} 成功 ({final_results['success_rate']:.1f}%)")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 包括的な本番環境テスト失敗: {e}")
            return {"error": str(e)}
    
    def _generate_performance_summary(self) -> Dict:
        """パフォーマンスサマリー生成"""
        summary = {
            "average_startup_time": 0,
            "average_memory_usage": 0,
            "average_cpu_usage": 0,
            "average_response_time": 0,
            "memory_leaks_detected": 0,
            "performance_issues": []
        }
        
        startup_times = []
        memory_usages = []
        cpu_usages = []
        response_times = []
        
        for result in self.test_results.values():
            if result.success:
                # 起動時間
                if "startup_time_seconds" in result.performance_metrics:
                    startup_times.append(result.performance_metrics["startup_time_seconds"])
                
                # メモリ使用量
                if "memory_usage_mb" in result.performance_metrics:
                    memory_usages.append(result.performance_metrics["memory_usage_mb"])
                
                # CPU使用量
                if "cpu_usage_percent" in result.performance_metrics:
                    cpu_usages.append(result.performance_metrics["cpu_usage_percent"])
                
                # 応答時間
                if "average_response_time_ms" in result.gui_responsiveness:
                    response_times.append(result.gui_responsiveness["average_response_time_ms"])
                
                # メモリリーク検出
                if "potential_memory_leak" in result.memory_usage:
                    if result.memory_usage["potential_memory_leak"]:
                        summary["memory_leaks_detected"] += 1
        
        # 平均値計算
        if startup_times:
            summary["average_startup_time"] = sum(startup_times) / len(startup_times)
        if memory_usages:
            summary["average_memory_usage"] = sum(memory_usages) / len(memory_usages)
        if cpu_usages:
            summary["average_cpu_usage"] = sum(cpu_usages) / len(cpu_usages)
        if response_times:
            summary["average_response_time"] = sum(response_times) / len(response_times)
        
        # パフォーマンス問題の検出
        if summary["average_startup_time"] > 10:
            summary["performance_issues"].append("起動時間が長い")
        if summary["average_memory_usage"] > self.test_config["memory_threshold_mb"]:
            summary["performance_issues"].append("メモリ使用量が高い")
        if summary["average_response_time"] > self.test_config["response_time_threshold_ms"]:
            summary["performance_issues"].append("応答時間が長い")
        
        return summary
    
    def _save_test_results(self, results: Dict):
        """テスト結果保存"""
        try:
            results_dir = Path("test_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"production_test_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ テスト結果保存: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ テスト結果保存失敗: {e}")

class PerformanceMonitor:
    """パフォーマンス監視クラス（改善版）"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.memory_history = []
        self.cpu_history = []
        self.disk_history = []
        self.start_time = None
        self.monitoring_interval = 0.1  # 100ms間隔で監視
        
        # メモリ使用量の閾値設定
        self.memory_threshold_mb = 1000  # 1GB
        self.cpu_threshold_percent = 80  # 80%
        self.disk_threshold_percent = 90  # 90%
    
    def start_monitoring(self):
        """監視開始"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.start_time = time.time()
            self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_performance(self):
        """パフォーマンス監視ループ"""
        while self.monitoring_active:
            try:
                # メモリ使用量監視
                memory_usage = self.get_memory_usage_mb()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'usage_mb': memory_usage
                })
                
                # CPU使用率監視
                cpu_usage = self.get_cpu_usage_percent()
                self.cpu_history.append({
                    'timestamp': time.time(),
                    'usage_percent': cpu_usage
                })
                
                # ディスク使用率監視
                disk_usage = self.get_disk_usage_percent()
                self.disk_history.append({
                    'timestamp': time.time(),
                    'usage_percent': disk_usage
                })
                
                # 閾値チェック
                self._check_thresholds(memory_usage, cpu_usage, disk_usage)
                
                # 履歴サイズ制限（最新1000件を保持）
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]
                if len(self.cpu_history) > 1000:
                    self.cpu_history = self.cpu_history[-1000:]
                if len(self.disk_history) > 1000:
                    self.disk_history = self.disk_history[-1000:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"パフォーマンス監視エラー: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_thresholds(self, memory_usage: float, cpu_usage: float, disk_usage: float):
        """閾値チェック"""
        if memory_usage > self.memory_threshold_mb:
            print(f"⚠️ メモリ使用量警告: {memory_usage:.1f}MB > {self.memory_threshold_mb}MB")
        
        if cpu_usage > self.cpu_threshold_percent:
            print(f"⚠️ CPU使用率警告: {cpu_usage:.1f}% > {self.cpu_threshold_percent}%")
        
        if disk_usage > self.disk_threshold_percent:
            print(f"⚠️ ディスク使用率警告: {disk_usage:.1f}% > {self.disk_threshold_percent}%")
    
    def get_memory_usage_mb(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_cpu_usage_percent(self) -> float:
        """現在のCPU使用率を取得（%）"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def get_disk_usage_percent(self) -> float:
        """現在のディスク使用率を取得（%）"""
        try:
            disk_usage = psutil.disk_usage('/')
            return (disk_usage.used / disk_usage.total) * 100
        except Exception:
            return 0.0
    
    def get_performance_summary(self) -> Dict:
        """パフォーマンスサマリーを取得"""
        if not self.memory_history:
            return {}
        
        memory_values = [h['usage_mb'] for h in self.memory_history]
        cpu_values = [h['usage_percent'] for h in self.cpu_history]
        disk_values = [h['usage_percent'] for h in self.disk_history]
        
        return {
            'memory': {
                'current_mb': memory_values[-1] if memory_values else 0,
                'average_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max_mb': max(memory_values) if memory_values else 0,
                'min_mb': min(memory_values) if memory_values else 0
            },
            'cpu': {
                'current_percent': cpu_values[-1] if cpu_values else 0,
                'average_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max_percent': max(cpu_values) if cpu_values else 0,
                'min_percent': min(cpu_values) if cpu_values else 0
            },
            'disk': {
                'current_percent': disk_values[-1] if disk_values else 0,
                'average_percent': sum(disk_values) / len(disk_values) if disk_values else 0,
                'max_percent': max(disk_values) if disk_values else 0,
                'min_percent': min(disk_values) if disk_values else 0
            },
            'monitoring_duration_seconds': time.time() - self.start_time if self.start_time else 0
        }
    
    def optimize_memory_usage(self):
        """メモリ使用量の最適化"""
        try:
            # ガベージコレクション実行
            gc.collect()
            
            # 履歴データのクリーンアップ
            if len(self.memory_history) > 500:
                self.memory_history = self.memory_history[-500:]
            if len(self.cpu_history) > 500:
                self.cpu_history = self.cpu_history[-500:]
            if len(self.disk_history) > 500:
                self.disk_history = self.disk_history[-500:]
            
            print("✅ メモリ使用量最適化完了")
            
        except Exception as e:
            print(f"❌ メモリ最適化エラー: {e}")

def main():
    """メイン実行関数（改善版）"""
    print("🚀 本番環境テストシステム起動")
    print(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 本番環境テストシステム初期化
    production_tester = ProductionEnvironmentTest()
    
    try:
        # パフォーマンス監視開始
        production_tester.performance_monitor.start_monitoring()
        
        # 包括的な本番環境テスト実行
        results = production_tester.run_comprehensive_production_test()
        
        # 結果表示
        print("\n" + "="*60)
        print("📊 本番環境テスト結果サマリー")
        print("="*60)
        
        if "error" in results:
            print(f"❌ テスト実行エラー: {results['error']}")
        else:
            print(f"✅ 総テスト数: {results['total_tests']}")
            print(f"✅ 成功テスト数: {results['successful_tests']}")
            print(f"❌ 失敗テスト数: {results['failed_tests']}")
            print(f"📈 成功率: {results['success_rate']:.1f}%")
            print(f"⏱️ 実行時間: {results['duration_seconds']:.2f}秒")
            
            # パフォーマンスサマリー
            if "performance_summary" in results:
                summary = results["performance_summary"]
                print(f"\n📊 パフォーマンスサマリー:")
                print(f"  平均起動時間: {summary['average_startup_time']:.2f}秒")
                print(f"  平均メモリ使用量: {summary['average_memory_usage']:.1f}MB")
                print(f"  平均CPU使用量: {summary['average_cpu_usage']:.1f}%")
                print(f"  平均応答時間: {summary['average_response_time']:.1f}ms")
                print(f"  メモリリーク検出: {summary['memory_leaks_detected']}件")
                
                if summary["performance_issues"]:
                    print(f"  ⚠️ パフォーマンス問題:")
                    for issue in summary["performance_issues"]:
                        print(f"    - {issue}")
            
            # パフォーマンス監視サマリー
            if "performance_monitor_summary" in results:
                monitor_summary = results["performance_monitor_summary"]
                if monitor_summary:
                    print(f"\n🖥️ システムリソース監視サマリー:")
                    print(f"  監視時間: {monitor_summary['monitoring_duration_seconds']:.1f}秒")
                    
                    if 'memory' in monitor_summary:
                        mem = monitor_summary['memory']
                        print(f"  メモリ使用量:")
                        print(f"    現在: {mem['current_mb']:.1f}MB")
                        print(f"    平均: {mem['average_mb']:.1f}MB")
                        print(f"    最大: {mem['max_mb']:.1f}MB")
                        print(f"    最小: {mem['min_mb']:.1f}MB")
                    
                    if 'cpu' in monitor_summary:
                        cpu = monitor_summary['cpu']
                        print(f"  CPU使用率:")
                        print(f"    現在: {cpu['current_percent']:.1f}%")
                        print(f"    平均: {cpu['average_percent']:.1f}%")
                        print(f"    最大: {cpu['max_percent']:.1f}%")
                        print(f"    最小: {cpu['min_percent']:.1f}%")
                    
                    if 'disk' in monitor_summary:
                        disk = monitor_summary['disk']
                        print(f"  ディスク使用率:")
                        print(f"    現在: {disk['current_percent']:.1f}%")
                        print(f"    平均: {disk['average_percent']:.1f}%")
                        print(f"    最大: {disk['max_percent']:.1f}%")
                        print(f"    最小: {disk['min_percent']:.1f}%")
            
            # テスト詳細
            if "test_details" in results:
                print(f"\n🔍 テスト詳細:")
                for test_name, test_detail in results["test_details"].items():
                    status = "✅" if test_detail.get("success", False) else "❌"
                    duration = test_detail.get("duration_seconds", 0)
                    print(f"  {status} {test_name}: {duration:.2f}秒")
                    
                    if not test_detail.get("success", False) and test_detail.get("error_message"):
                        print(f"    エラー: {test_detail['error_message']}")
        
        print("="*60)
        
        # 改善提案
        if "success_rate" in results and results["success_rate"] < 100:
            print(f"\n💡 改善提案:")
            print(f"  - テスト成功率が{results['success_rate']:.1f}%のため、エラーハンドリングの強化を推奨")
            print(f"  - データ処理エラーの詳細調査が必要")
            print(f"  - メモリ使用量の最適化を検討")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        production_tester.performance_monitor.stop_monitoring()
        
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        traceback.print_exc()
        production_tester.performance_monitor.stop_monitoring()
        
    finally:
        # メモリ最適化
        production_tester.performance_monitor.optimize_memory_usage()
        print("✅ テストシステム終了")

if __name__ == "__main__":
    main() 