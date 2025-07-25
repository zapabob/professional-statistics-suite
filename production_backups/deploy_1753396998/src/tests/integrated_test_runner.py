#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated Test Runner
統合テストランナー

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import time
import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import traceback

# テストモジュールのインポート
from src.tests.e2e_test_automation import E2ETestAutomation
from src.tests.gui_button_test_automation import GUIButtonTestAutomation
from src.tests.production_environment_test import ProductionEnvironmentTest

class IntegratedTestRunner:
    """統合テストランナー"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.test_systems = {}
        
        # ログ設定
        self._setup_logging()
        
        # テストシステム初期化
        self._initialize_test_systems()
    
    def _setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"integrated_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_test_systems(self):
        """テストシステム初期化"""
        try:
            self.test_systems["e2e"] = E2ETestAutomation()
            self.test_systems["gui_button"] = GUIButtonTestAutomation()
            self.test_systems["production"] = ProductionEnvironmentTest()
            
            self.logger.info("✅ テストシステム初期化完了")
            
        except Exception as e:
            self.logger.error(f"❌ テストシステム初期化失敗: {e}")
    
    async def run_e2e_tests(self) -> Dict:
        """E2Eテスト実行"""
        self.logger.info("🚀 E2Eテスト実行開始")
        
        try:
            results = await self.test_systems["e2e"].run_comprehensive_e2e_test()
            self.test_results["e2e"] = results
            
            self.logger.info("✅ E2Eテスト実行完了")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ E2Eテスト実行失敗: {e}")
            return {"error": str(e)}
    
    def run_gui_button_tests(self) -> Dict:
        """GUIボタンテスト実行"""
        self.logger.info("🚀 GUIボタンテスト実行開始")
        
        try:
            results = self.test_systems["gui_button"].run_comprehensive_button_test()
            self.test_results["gui_button"] = results
            
            self.logger.info("✅ GUIボタンテスト実行完了")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ GUIボタンテスト実行失敗: {e}")
            return {"error": str(e)}
    
    def run_production_tests(self) -> Dict:
        """本番環境テスト実行"""
        self.logger.info("🚀 本番環境テスト実行開始")
        
        try:
            results = self.test_systems["production"].run_comprehensive_production_test()
            self.test_results["production"] = results
            
            self.logger.info("✅ 本番環境テスト実行完了")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 本番環境テスト実行失敗: {e}")
            return {"error": str(e)}
    
    async def run_all_tests(self) -> Dict:
        """全テスト実行"""
        self.logger.info("🚀 統合テスト実行開始")
        
        self.start_time = datetime.now()
        
        try:
            # 並列実行でテストを開始
            tasks = [
                self.run_e2e_tests(),
                asyncio.create_task(self._run_sync_test(self.run_gui_button_tests)),
                asyncio.create_task(self._run_sync_test(self.run_production_tests))
            ]
            
            # 全テスト完了を待機
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を整理
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ テスト実行エラー: {result}")
                    self.test_results[f"test_{i}"] = {"error": str(result)}
            
            self.end_time = datetime.now()
            
            # 統合結果生成
            integrated_results = self._generate_integrated_results()
            
            # 結果保存
            self._save_integrated_results(integrated_results)
            
            self.logger.info("✅ 統合テスト実行完了")
            return integrated_results
            
        except Exception as e:
            self.logger.error(f"❌ 統合テスト実行失敗: {e}")
            return {"error": str(e)}
    
    async def _run_sync_test(self, test_func):
        """同期テストを非同期で実行"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, test_func)
    
    def _generate_integrated_results(self) -> Dict:
        """統合結果生成"""
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        test_details = {}
        
        # 各テストシステムの結果を集計
        for system_name, results in self.test_results.items():
            if "error" in results:
                failed_tests += 1
                test_details[system_name] = {
                    "status": "failed",
                    "error": results["error"]
                }
            else:
                if "total_tests" in results:
                    total_tests += results["total_tests"]
                    successful_tests += results.get("successful_tests", 0)
                    failed_tests += results.get("failed_tests", 0)
                
                test_details[system_name] = {
                    "status": "success",
                    "results": results
                }
        
        # 実行時間計算
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        # 成功率計算
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "execution_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": duration,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "test_systems": test_details,
            "overall_status": "success" if success_rate >= 80 else "warning" if success_rate >= 60 else "failed"
        }
    
    def _save_integrated_results(self, results: Dict):
        """統合結果保存"""
        try:
            results_dir = Path("test_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"integrated_test_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # HTMLレポート生成
            html_report = self._generate_html_report(results)
            html_file = results_dir / f"integrated_test_report_{timestamp}.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            self.logger.info(f"✅ 統合結果保存: {results_file}")
            self.logger.info(f"✅ HTMLレポート生成: {html_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 統合結果保存失敗: {e}")
    
    def _generate_html_report(self, results: Dict) -> str:
        """HTMLレポート生成"""
        html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>統合テストレポート</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .summary {
            padding: 30px;
            border-bottom: 1px solid #eee;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .summary-card .label {
            color: #666;
            font-size: 0.9em;
        }
        .test-details {
            padding: 30px;
        }
        .test-section {
            margin-bottom: 30px;
        }
        .test-section h3 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .test-result {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            border-left: 4px solid #28a745;
        }
        .test-result.failed {
            border-left-color: #dc3545;
            background: #f8d7da;
        }
        .test-result.warning {
            border-left-color: #ffc107;
            background: #fff3cd;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        .status-warning {
            background: #fff3cd;
            color: #856404;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        .progress-fill.warning {
            background: linear-gradient(90deg, #ffc107, #fd7e14);
        }
        .progress-fill.failed {
            background: linear-gradient(90deg, #dc3545, #e83e8c);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 統合テストレポート</h1>
            <p>Professional Statistics Suite - 包括的テスト結果</p>
        </div>
        
        <div class="summary">
            <h2>📊 テストサマリー</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>総テスト数</h3>
                    <div class="value">{total_tests}</div>
                    <div class="label">実行されたテストの総数</div>
                </div>
                <div class="summary-card">
                    <h3>成功テスト</h3>
                    <div class="value">{successful_tests}</div>
                    <div class="label">正常に完了したテスト</div>
                </div>
                <div class="summary-card">
                    <h3>失敗テスト</h3>
                    <div class="value">{failed_tests}</div>
                    <div class="label">失敗したテスト</div>
                </div>
                <div class="summary-card">
                    <h3>成功率</h3>
                    <div class="value">{success_rate:.1f}%</div>
                    <div class="label">テスト成功率</div>
                </div>
                <div class="summary-card">
                    <h3>実行時間</h3>
                    <div class="value">{duration:.1f}s</div>
                    <div class="label">総実行時間</div>
                </div>
                <div class="summary-card">
                    <h3>全体ステータス</h3>
                    <div class="value">
                        <span class="status-badge status-{overall_status}">{overall_status}</span>
                    </div>
                    <div class="label">テスト全体の状態</div>
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill {progress_class}" style="width: {success_rate}%"></div>
            </div>
        </div>
        
        <div class="test-details">
            <h2>🔍 詳細結果</h2>
            
            {test_details_html}
        </div>
        
        <div class="footer">
            <p>生成日時: {timestamp}</p>
            <p>Professional Statistics Suite - 統合テストシステム</p>
        </div>
    </div>
</body>
</html>
        """
        
        # データ準備
        exec_info = results["execution_info"]
        overall_status = results["overall_status"]
        
        # プログレスバーのクラス決定
        progress_class = "success" if overall_status == "success" else "warning" if overall_status == "warning" else "failed"
        
        # テスト詳細HTML生成
        test_details_html = ""
        for system_name, system_result in results["test_systems"].items():
            status_class = system_result["status"]
            test_details_html += f"""
            <div class="test-section">
                <h3>🧪 {system_name.replace('_', ' ').title()} テスト</h3>
                <div class="test-result {status_class}">
                    <strong>ステータス:</strong> 
                    <span class="status-badge status-{status_class}">{status_class}</span>
            """
            
            if status_class == "success":
                system_data = system_result["results"]
                if "total_tests" in system_data:
                    test_details_html += f"""
                    <br><strong>テスト数:</strong> {system_data.get('total_tests', 0)}
                    <br><strong>成功:</strong> {system_data.get('successful_tests', 0)}
                    <br><strong>失敗:</strong> {system_data.get('failed_tests', 0)}
                    <br><strong>成功率:</strong> {system_data.get('success_rate', 0):.1f}%
                    """
            else:
                test_details_html += f"<br><strong>エラー:</strong> {system_result.get('error', 'Unknown error')}"
            
            test_details_html += "</div></div>"
        
        # HTML生成
        return html_template.format(
            total_tests=exec_info["total_tests"],
            successful_tests=exec_info["successful_tests"],
            failed_tests=exec_info["failed_tests"],
            success_rate=exec_info["success_rate"],
            duration=exec_info["duration_seconds"],
            overall_status=overall_status,
            progress_class=progress_class,
            test_details_html=test_details_html,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def cleanup(self):
        """クリーンアップ"""
        try:
            for system in self.test_systems.values():
                if hasattr(system, 'cleanup'):
                    system.cleanup()
            
            self.logger.info("✅ 統合テストランナークリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"❌ クリーンアップ失敗: {e}")

async def main():
    """メイン実行関数"""
    print("🚀 統合テストランナー起動")
    
    # 統合テストランナー初期化
    runner = IntegratedTestRunner()
    
    try:
        # 全テスト実行
        results = await runner.run_all_tests()
        
        # 結果表示
        print("\n" + "="*60)
        print("📊 統合テスト結果サマリー")
        print("="*60)
        
        if "error" in results:
            print(f"❌ テスト実行エラー: {results['error']}")
        else:
            exec_info = results["execution_info"]
            print(f"✅ 総テスト数: {exec_info['total_tests']}")
            print(f"✅ 成功テスト数: {exec_info['successful_tests']}")
            print(f"❌ 失敗テスト数: {exec_info['failed_tests']}")
            print(f"📈 成功率: {exec_info['success_rate']:.1f}%")
            print(f"⏱️ 実行時間: {exec_info['duration_seconds']:.2f}秒")
            print(f"🎯 全体ステータス: {results['overall_status']}")
            
            # 各テストシステムの結果
            print("\n📋 テストシステム詳細:")
            for system_name, system_result in results["test_systems"].items():
                status = system_result["status"]
                if status == "success":
                    system_data = system_result["results"]
                    if "total_tests" in system_data:
                        print(f"  ✅ {system_name}: {system_data.get('successful_tests', 0)}/{system_data.get('total_tests', 0)} 成功")
                    else:
                        print(f"  ✅ {system_name}: 成功")
                else:
                    print(f"  ❌ {system_name}: 失敗 - {system_result.get('error', 'Unknown error')}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        traceback.print_exc()
        
    finally:
        runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 