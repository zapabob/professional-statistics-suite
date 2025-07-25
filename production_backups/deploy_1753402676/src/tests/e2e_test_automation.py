#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E2E Test Automation System
E2Eテスト自動化システム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import time
import json
import pickle
import signal
import threading
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging
import traceback

# GUI関連
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np

# テスト関連
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available. Install with: pip install playwright")

# プロジェクト固有のインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import SPSSGradeConfig
from src.gui.professional_statistics_gui import ProfessionalStatisticsGUI
from src.gui.unified_ai_landing_gui import UnifiedAILandingGUI
from src.gui.kiro_integrated_gui import KiroIntegratedGUI

class CheckpointManager:
    """自動チェックポイント保存システム"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backups = 10
        self.checkpoint_interval = 300  # 5分間隔
        self.last_checkpoint = time.time()
        self.running = True
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
        
        # 定期保存スレッド開始
        self.checkpoint_thread = threading.Thread(target=self._periodic_checkpoint, daemon=True)
        self.checkpoint_thread.start()
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー: 緊急保存"""
        print(f"\n🛡️ 緊急保存開始 (シグナル: {signum})")
        self.emergency_save()
        sys.exit(0)
    
    def _periodic_checkpoint(self):
        """定期チェックポイント保存"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_checkpoint >= self.checkpoint_interval:
                    self.save_checkpoint("periodic")
                    self.last_checkpoint = current_time
                time.sleep(10)  # 10秒間隔でチェック
            except Exception as e:
                print(f"⚠️ 定期チェックポイントエラー: {e}")
    
    def save_checkpoint(self, checkpoint_type: str = "manual"):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "checkpoint_type": checkpoint_type,
                "test_state": self._get_current_state(),
                "metadata": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "working_directory": os.getcwd()
                }
            }
            
            # JSON保存
            json_file = self.checkpoint_dir / f"checkpoint_{self.session_id}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            # Pickle保存（オブジェクト状態）
            pickle_file = self.checkpoint_dir / f"checkpoint_{self.session_id}_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # バックアップ管理
            self._manage_backups()
            
            print(f"✅ チェックポイント保存完了: {checkpoint_type} - {timestamp}")
            return True
            
        except Exception as e:
            print(f"❌ チェックポイント保存失敗: {e}")
            return False
    
    def emergency_save(self):
        """緊急保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_data = {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "checkpoint_type": "emergency",
                "test_state": self._get_current_state(),
                "error_info": {
                    "traceback": traceback.format_exc(),
                    "memory_usage": self._get_memory_usage()
                }
            }
            
            # 緊急保存ファイル
            emergency_file = self.checkpoint_dir / f"emergency_{self.session_id}_{timestamp}.json"
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_data, f, ensure_ascii=False, indent=2)
            
            print(f"🛡️ 緊急保存完了: {emergency_file}")
            return True
            
        except Exception as e:
            print(f"❌ 緊急保存失敗: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """チェックポイント読み込み"""
        try:
            checkpoint_path = self.checkpoint_dir / checkpoint_file
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"❌ チェックポイント読み込み失敗: {e}")
            return None
    
    def _get_current_state(self) -> Dict:
        """現在の状態取得"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_tests": getattr(self, 'active_tests', []),
            "test_results": getattr(self, 'test_results', {}),
            "gui_state": getattr(self, 'gui_state', {}),
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict:
        """メモリ使用量取得"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def _manage_backups(self):
        """バックアップ管理"""
        try:
            # 古いバックアップを削除
            backup_files = list(self.backup_dir.glob("*.json"))
            if len(backup_files) > self.max_backups:
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in backup_files[:-self.max_backups]:
                    old_file.unlink()
                    print(f"🗑️ 古いバックアップ削除: {old_file.name}")
        except Exception as e:
            print(f"⚠️ バックアップ管理エラー: {e}")
    
    def cleanup(self):
        """クリーンアップ"""
        self.running = False
        if hasattr(self, 'checkpoint_thread'):
            self.checkpoint_thread.join(timeout=5)

class E2ETestAutomation:
    """E2Eテスト自動化システム"""
    
    def __init__(self, config: Optional[SPSSGradeConfig] = None):
        self.config = config or SPSSGradeConfig()
        self.checkpoint_manager = CheckpointManager()
        self.test_results = {}
        self.gui_instances = {}
        self.browser = None
        self.page = None
        self.test_log = []
        
        # ログ設定
        self._setup_logging()
        
        # テストデータ
        self.test_data = self._create_test_data()
    
    def _setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_test_data(self) -> Dict[str, pd.DataFrame]:
        """テストデータ作成"""
        return {
            "basic": pd.DataFrame({
                'group': ['A', 'A', 'B', 'B', 'C', 'C'],
                'score': [85, 90, 78, 82, 88, 92],
                'age': [25, 30, 28, 35, 22, 27]
            }),
            "regression": pd.DataFrame({
                'x': np.random.normal(0, 1, 100),
                'y': np.random.normal(0, 1, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            }),
            "timeseries": pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100, freq='D'),
                'value': np.random.normal(100, 10, 100).cumsum()
            })
        }
    
    async def setup_browser(self):
        """ブラウザセットアップ"""
        if not PLAYWRIGHT_AVAILABLE:
            self.logger.warning("Playwright not available. Skipping browser tests.")
            return False
        
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,  # GUIテスト用にヘッドレス無効
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            
            self.logger.info("✅ ブラウザセットアップ完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ブラウザセットアップ失敗: {e}")
            return False
    
    async def cleanup_browser(self):
        """ブラウザクリーンアップ"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            
            self.logger.info("✅ ブラウザクリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"❌ ブラウザクリーンアップ失敗: {e}")
    
    def test_gui_button_functionality(self, gui_class, test_name: str):
        """GUIボタン機能テスト"""
        self.logger.info(f"🧪 GUIボタンテスト開始: {test_name}")
        
        try:
            # GUIインスタンス作成
            root = tk.Tk()
            root.withdraw()  # バックグラウンド実行
            
            gui_instance = gui_class(root)
            self.gui_instances[test_name] = gui_instance
            
            # ボタン要素の取得とテスト
            buttons = self._find_buttons(gui_instance)
            
            test_results = {
                "total_buttons": len(buttons),
                "tested_buttons": 0,
                "successful_clicks": 0,
                "failed_clicks": 0,
                "button_details": []
            }
            
            for button_name, button_widget in buttons.items():
                try:
                    # ボタンクリックテスト
                    button_widget.invoke()
                    test_results["successful_clicks"] += 1
                    test_results["button_details"].append({
                        "name": button_name,
                        "status": "success",
                        "text": button_widget.cget("text") if hasattr(button_widget, 'cget') else "N/A"
                    })
                    
                except Exception as e:
                    test_results["failed_clicks"] += 1
                    test_results["button_details"].append({
                        "name": button_name,
                        "status": "failed",
                        "error": str(e)
                    })
                
                test_results["tested_buttons"] += 1
            
            # 結果記録
            self.test_results[test_name] = test_results
            self.logger.info(f"✅ {test_name} ボタンテスト完了: {test_results['successful_clicks']}/{test_results['tested_buttons']} 成功")
            
            # クリーンアップ
            root.destroy()
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} ボタンテスト失敗: {e}")
            return {"error": str(e)}
    
    def _find_buttons(self, widget) -> Dict[str, tk.Widget]:
        """ウィジェット内のボタンを再帰的に検索"""
        buttons = {}
        
        def search_buttons(w):
            if isinstance(w, (tk.Button, ttk.Button)):
                button_name = w.cget("text") if hasattr(w, 'cget') else str(w)
                buttons[button_name] = w
            
            # 子ウィジェットを再帰的に検索
            for child in w.winfo_children():
                search_buttons(child)
        
        search_buttons(widget)
        return buttons
    
    async def test_web_interface(self, url: str, test_name: str):
        """Webインターフェーステスト"""
        if not self.page:
            self.logger.error("❌ ブラウザがセットアップされていません")
            return False
        
        self.logger.info(f"🧪 Webインターフェーステスト開始: {test_name}")
        
        try:
            # ページ読み込み
            await self.page.goto(url)
            await self.page.wait_for_load_state('networkidle')
            
            # ボタン要素の検索とテスト
            buttons = await self.page.query_selector_all('button, input[type="button"], input[type="submit"]')
            
            test_results = {
                "total_buttons": len(buttons),
                "tested_buttons": 0,
                "successful_clicks": 0,
                "failed_clicks": 0,
                "button_details": []
            }
            
            for i, button in enumerate(buttons):
                try:
                    # ボタンテキスト取得
                    button_text = await button.text_content() or f"Button_{i}"
                    
                    # ボタンクリックテスト
                    await button.click()
                    await self.page.wait_for_timeout(1000)  # 1秒待機
                    
                    test_results["successful_clicks"] += 1
                    test_results["button_details"].append({
                        "index": i,
                        "text": button_text,
                        "status": "success"
                    })
                    
                except Exception as e:
                    test_results["failed_clicks"] += 1
                    test_results["button_details"].append({
                        "index": i,
                        "text": button_text if 'button_text' in locals() else f"Button_{i}",
                        "status": "failed",
                        "error": str(e)
                    })
                
                test_results["tested_buttons"] += 1
            
            # スクリーンショット保存
            screenshot_path = f"test_screenshots/{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            os.makedirs("test_screenshots", exist_ok=True)
            await self.page.screenshot(path=screenshot_path)
            
            self.test_results[test_name] = test_results
            self.logger.info(f"✅ {test_name} Webテスト完了: {test_results['successful_clicks']}/{test_results['tested_buttons']} 成功")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} Webテスト失敗: {e}")
            return {"error": str(e)}
    
    def test_data_processing_pipeline(self, test_name: str):
        """データ処理パイプラインテスト"""
        self.logger.info(f"🧪 データ処理パイプラインテスト開始: {test_name}")
        
        try:
            test_data = self.test_data["basic"].copy()
            
            # データ前処理テスト
            from data.data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            
            # 欠損値処理
            processed_data = preprocessor.handle_missing_values(test_data)
            
            # 外れ値検出
            outliers = preprocessor.detect_outliers(processed_data)
            
            # 統計分析
            from statistics.advanced_statistics import AdvancedStatistics
            stats = AdvancedStatistics()
            basic_stats = stats.calculate_basic_statistics(processed_data)
            
            test_results = {
                "original_size": test_data.shape,
                "processed_size": processed_data.shape,
                "outliers_detected": len(outliers),
                "basic_stats_calculated": len(basic_stats),
                "pipeline_status": "success"
            }
            
            self.test_results[test_name] = test_results
            self.logger.info(f"✅ {test_name} データ処理パイプラインテスト完了")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} データ処理パイプラインテスト失敗: {e}")
            return {"error": str(e)}
    
    async def test_ai_integration_pipeline(self, test_name: str):
        """AI統合パイプラインテスト"""
        self.logger.info(f"🧪 AI統合パイプラインテスト開始: {test_name}")
        
        try:
            from ai.ai_integration import AIOrchestrator
            
            # AIオーケストレーター初期化
            orchestrator = AIOrchestrator()
            
            # テストクエリ
            test_query = "このデータセットの基本統計を計算してください"
            test_data = self.test_data["basic"]
            
            # AI分析実行（AnalysisContextを作成してprocess_user_queryを使用）
            from src.ai.ai_integration import AnalysisContext
            context = AnalysisContext(
                user_id="test_user",
                session_id="test_session",
                data_fingerprint="test_data",
                analysis_history=[]
            )
            result = await orchestrator.process_user_query(test_query, context, test_data)
            
            test_results = {
                "query_processed": True,
                "ai_response_received": result is not None,
                "response_length": len(str(result)) if result else 0,
                "pipeline_status": "success"
            }
            
            self.test_results[test_name] = test_results
            self.logger.info(f"✅ {test_name} AI統合パイプラインテスト完了")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} AI統合パイプラインテスト失敗: {e}")
            return {"error": str(e)}
    
    async def run_comprehensive_e2e_test(self):
        """包括的なE2Eテスト実行"""
        self.logger.info("🚀 包括的なE2Eテスト開始")
        
        start_time = time.time()
        
        try:
            # ブラウザセットアップ
            browser_ready = await self.setup_browser()
            
            # GUIボタンテスト
            gui_tests = [
                (ProfessionalStatisticsGUI, "ProfessionalStatisticsGUI_Buttons"),
                (UnifiedAILandingGUI, "UnifiedAILandingGUI_Buttons"),
                (KiroIntegratedGUI, "KiroIntegratedGUI_Buttons")
            ]
            
            for gui_class, test_name in gui_tests:
                self.test_gui_button_functionality(gui_class, test_name)
                self.checkpoint_manager.save_checkpoint(f"gui_test_{test_name}")
            
            # Webインターフェーステスト（ローカルサーバーがある場合）
            if browser_ready:
                await self.test_web_interface("http://localhost:8000", "LocalWebInterface")
                self.checkpoint_manager.save_checkpoint("web_interface_test")
            
            # データ処理パイプラインテスト
            self.test_data_processing_pipeline("DataProcessingPipeline")
            self.checkpoint_manager.save_checkpoint("data_pipeline_test")
            
            # AI統合パイプラインテスト
            await self.test_ai_integration_pipeline("AIIntegrationPipeline")
            self.checkpoint_manager.save_checkpoint("ai_pipeline_test")
            
            # ブラウザクリーンアップ
            if browser_ready:
                await self.cleanup_browser()
            
            # 最終結果集計
            total_tests = len(self.test_results)
            successful_tests = sum(1 for result in self.test_results.values() if "error" not in result)
            
            end_time = time.time()
            duration = end_time - start_time
            
            final_results = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "duration_seconds": duration,
                "test_details": self.test_results
            }
            
            # 結果保存
            self._save_test_results(final_results)
            
            self.logger.info(f"✅ E2Eテスト完了: {successful_tests}/{total_tests} 成功 ({final_results['success_rate']:.1f}%)")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ E2Eテスト失敗: {e}")
            self.checkpoint_manager.emergency_save()
            return {"error": str(e)}
    
    def _save_test_results(self, results: Dict):
        """テスト結果保存"""
        try:
            results_dir = Path("test_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"e2e_test_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ テスト結果保存: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ テスト結果保存失敗: {e}")
    
    def cleanup(self):
        """クリーンアップ"""
        try:
            self.checkpoint_manager.cleanup()
            
            # GUIインスタンスのクリーンアップ
            for gui_instance in self.gui_instances.values():
                if hasattr(gui_instance, 'destroy'):
                    gui_instance.destroy()
            
            self.logger.info("✅ E2Eテスト自動化システムクリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"❌ クリーンアップ失敗: {e}")

def main():
    """メイン実行関数"""
    print("🚀 E2Eテスト自動化システム起動")
    
    # テスト自動化システム初期化
    e2e_tester = E2ETestAutomation()
    
    try:
        # 包括的なE2Eテスト実行
        results = asyncio.run(e2e_tester.run_comprehensive_e2e_test())
        
        # 結果表示
        print("\n" + "="*50)
        print("📊 E2Eテスト結果サマリー")
        print("="*50)
        
        if "error" in results:
            print(f"❌ テスト実行エラー: {results['error']}")
        else:
            print(f"✅ 総テスト数: {results['total_tests']}")
            print(f"✅ 成功テスト数: {results['successful_tests']}")
            print(f"❌ 失敗テスト数: {results['failed_tests']}")
            print(f"📈 成功率: {results['success_rate']:.1f}%")
            print(f"⏱️ 実行時間: {results['duration_seconds']:.2f}秒")
            
            # 詳細結果
            print("\n📋 詳細結果:")
            for test_name, test_result in results['test_details'].items():
                if "error" in test_result:
                    print(f"  ❌ {test_name}: {test_result['error']}")
                else:
                    print(f"  ✅ {test_name}: 成功")
        
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        e2e_tester.checkpoint_manager.emergency_save()
        
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        e2e_tester.checkpoint_manager.emergency_save()
        
    finally:
        e2e_tester.cleanup()

if __name__ == "__main__":
    main() 