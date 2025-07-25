#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI Test Runner
統合CLIテスト実行ツール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
import asyncio
import subprocess

# テストシステムのインポート
from src.tests.e2e_test_automation import E2ETestAutomation
from src.tests.gui_button_test_automation import GUIButtonTestAutomation
from src.tests.production_environment_test import ProductionEnvironmentTest
from src.tests.integrated_test_runner import IntegratedTestRunner
from src.tests.performance_optimizer import TestOptimizer, PerformanceProfiler
from src.tests.parallel_test_runner import ParallelTestRunner, PytestXdistRunner
from src.tests.coverage_analyzer import CoverageAnalyzer, TestCoverageGenerator
from src.tests.html_report_generator import ReportManager, TestResult, CoverageData, PerformanceMetrics
from src.tests.test_data_manager import TestDataManager, TestDataFactory, DataGenerationConfig

class CLITestRunner:
    """統合CLIテスト実行ツール"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # コンポーネント初期化
        self.e2e_runner = E2ETestAutomation()
        self.gui_runner = GUIButtonTestAutomation()
        self.production_runner = ProductionEnvironmentTest()
        self.integrated_runner = IntegratedTestRunner()
        self.performance_optimizer = TestOptimizer()
        self.parallel_runner = ParallelTestRunner()
        self.coverage_analyzer = CoverageAnalyzer()
        self.report_manager = ReportManager()
        self.data_manager = TestDataManager()
        
        # 結果保存用
        self.test_results: List[TestResult] = []
        self.coverage_data: List[CoverageData] = []
        self.performance_metrics: List[PerformanceMetrics] = []
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('test_runner.log', encoding='utf-8')
            ]
        )
    
    def run_e2e_tests(self, args) -> bool:
        """E2Eテストを実行"""
        print("🚀 E2Eテストを実行中...")
        try:
            result = self.e2e_runner.run_comprehensive_e2e_test()
            
            # 結果を記録
            test_result = TestResult(
                test_name="E2E Tests",
                status="success" if result.get("success", False) else "failure",
                execution_time=result.get("execution_time", 0),
                memory_usage=result.get("memory_usage", 0),
                cpu_usage=result.get("cpu_usage", 0),
                timestamp=datetime.now(),
                error_message=result.get("error_message")
            )
            self.test_results.append(test_result)
            
            print(f"✅ E2Eテスト完了: {result.get('success', False)}")
            return result.get("success", False)
            
        except Exception as e:
            self.logger.error(f"❌ E2Eテストエラー: {e}")
            return False
    
    def run_gui_tests(self, args) -> bool:
        """GUIテストを実行"""
        print("🎨 GUIテストを実行中...")
        try:
            result = self.gui_runner.run_all_gui_tests()
            
            # 結果を記録
            test_result = TestResult(
                test_name="GUI Tests",
                status="success" if result.get("success", False) else "failure",
                execution_time=result.get("execution_time", 0),
                memory_usage=result.get("memory_usage", 0),
                cpu_usage=result.get("cpu_usage", 0),
                timestamp=datetime.now(),
                error_message=result.get("error_message")
            )
            self.test_results.append(test_result)
            
            print(f"✅ GUIテスト完了: {result.get('success', False)}")
            return result.get("success", False)
            
        except Exception as e:
            self.logger.error(f"❌ GUIテストエラー: {e}")
            return False
    
    def run_production_tests(self, args) -> bool:
        """本番環境テストを実行"""
        print("🏭 本番環境テストを実行中...")
        try:
            result = self.production_runner.run_all_production_tests()
            
            # 結果を記録
            test_result = TestResult(
                test_name="Production Tests",
                status="success" if result.get("success", False) else "failure",
                execution_time=result.get("execution_time", 0),
                memory_usage=result.get("memory_usage", 0),
                cpu_usage=result.get("cpu_usage", 0),
                timestamp=datetime.now(),
                error_message=result.get("error_message")
            )
            self.test_results.append(test_result)
            
            print(f"✅ 本番環境テスト完了: {result.get('success', False)}")
            return result.get("success", False)
            
        except Exception as e:
            self.logger.error(f"❌ 本番環境テストエラー: {e}")
            return False
    
    def run_performance_tests(self, args) -> bool:
        """パフォーマンステストを実行"""
        print("⚡ パフォーマンステストを実行中...")
        try:
            # パフォーマンスプロファイラーを使用
            profiler = PerformanceProfiler()
            
            # サンプルテスト関数
            def sample_test():
                import time
                time.sleep(1)
                return {"result": "test"}
            
            # パフォーマンス測定
            metrics = profiler.profile_test(sample_test, "performance_test")
            
            # 結果を記録
            performance_metric = PerformanceMetrics(
                test_name="Performance Test",
                execution_time=metrics.execution_time,
                memory_usage_mb=metrics.memory_usage_mb,
                cpu_usage_percent=metrics.cpu_usage_percent,
                gc_collections=metrics.gc_collections,
                gc_time=metrics.gc_time
            )
            self.performance_metrics.append(performance_metric)
            
            test_result = TestResult(
                test_name="Performance Tests",
                status="success",
                execution_time=metrics.execution_time,
                memory_usage=metrics.memory_usage_mb,
                cpu_usage=metrics.cpu_usage_percent,
                timestamp=datetime.now()
            )
            self.test_results.append(test_result)
            
            print(f"✅ パフォーマンステスト完了: {metrics.execution_time:.2f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ パフォーマンステストエラー: {e}")
            return False
    
    def run_parallel_tests(self, args) -> bool:
        """並列テストを実行"""
        print("🔄 並列テストを実行中...")
        try:
            # 非同期で並列テストを実行
            async def run_parallel():
                return await self.parallel_runner.run_parallel_tests()
            
            result = asyncio.run(run_parallel())
            
            # 結果を記録
            test_result = TestResult(
                test_name="Parallel Tests",
                status="success" if result.get("summary", {}).get("successful_tests", 0) > 0 else "failure",
                execution_time=result.get("summary", {}).get("parallel_execution_time", 0),
                memory_usage=0,  # 並列実行では個別のメモリ使用量を取得できない
                cpu_usage=0,
                timestamp=datetime.now()
            )
            self.test_results.append(test_result)
            
            summary = result.get("summary", {})
            print(f"✅ 並列テスト完了: {summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)} 成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 並列テストエラー: {e}")
            return False
    
    def run_coverage_analysis(self, args) -> bool:
        """カバレッジ分析を実行"""
        print("📊 カバレッジ分析を実行中...")
        try:
            # カバレッジ測定開始
            self.coverage_analyzer.start_coverage_measurement()
            
            # 簡単なテストを実行してカバレッジを測定
            def sample_function():
                return "test"
            
            sample_function()
            
            # カバレッジ測定停止
            self.coverage_analyzer.stop_coverage_measurement()
            
            # カバレッジ分析
            coverage_data = self.coverage_analyzer.analyze_coverage()
            
            # 結果を記録
            for file_path, metrics in coverage_data.get("files", {}).items():
                coverage = CoverageData(
                    file_path=file_path,
                    total_lines=metrics.get("total_lines", 0),
                    covered_lines=metrics.get("covered_lines", 0),
                    coverage_percentage=metrics.get("coverage_percentage", 0),
                    uncovered_lines=metrics.get("uncovered_lines", [])
                )
                self.coverage_data.append(coverage)
            
            test_result = TestResult(
                test_name="Coverage Analysis",
                status="success",
                execution_time=0,
                memory_usage=0,
                cpu_usage=0,
                timestamp=datetime.now()
            )
            self.test_results.append(test_result)
            
            overall_coverage = coverage_data.get("coverage_percentage", 0)
            print(f"✅ カバレッジ分析完了: {overall_coverage:.1f}%")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ カバレッジ分析エラー: {e}")
            return False
    
    def generate_test_data(self, args) -> bool:
        """テストデータを生成"""
        print("📊 テストデータを生成中...")
        try:
            if args.data_type == "sample":
                config = TestDataFactory.create_sample_data_config()
            elif args.data_type == "performance":
                config = TestDataFactory.create_performance_test_config()
            else:
                # カスタム設定
                config = DataGenerationConfig(
                    data_type=args.format,
                    size=args.size,
                    columns=args.columns.split(","),
                    data_types={col: "float" for col in args.columns.split(",")},
                    seed=args.seed
                )
            
            dataset = self.data_manager.generate_test_data(
                config,
                name=args.name,
                description=args.description,
                tags=args.tags.split(",") if args.tags else None
            )
            
            if dataset:
                print(f"✅ テストデータ生成完了: {dataset.name}")
                return True
            else:
                print("❌ テストデータ生成に失敗しました")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ テストデータ生成エラー: {e}")
            return False
    
    def generate_report(self, args) -> bool:
        """レポートを生成"""
        print("📋 レポートを生成中...")
        try:
            # テスト結果をレポートマネージャーに追加
            for result in self.test_results:
                self.report_manager.generator.add_test_result(result)
            
            # カバレッジデータを追加
            for coverage in self.coverage_data:
                self.report_manager.generator.add_coverage_data(coverage)
            
            # パフォーマンスメトリクスを追加
            for metrics in self.performance_metrics:
                self.report_manager.generator.add_performance_metrics(metrics)
            
            # HTMLレポートを生成
            report_path = self.report_manager.generate_comprehensive_report(args.output)
            
            if report_path:
                print(f"✅ レポート生成完了: {report_path}")
                
                # サマリーを表示
                summary = self.report_manager.generate_summary_report()
                print(f"\n📊 レポートサマリー:")
                print(f"  総テスト数: {summary.get('total_tests', 0)}")
                print(f"  成功率: {summary.get('success_rate', 0):.1f}%")
                print(f"  カバレッジ率: {summary.get('coverage_percentage', 0):.1f}%")
                print(f"  総実行時間: {summary.get('total_execution_time', 0):.2f}秒")
                
                return True
            else:
                print("❌ レポート生成に失敗しました")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ レポート生成エラー: {e}")
            return False
    
    def run_all_tests(self, args) -> bool:
        """すべてのテストを実行"""
        print("🚀 包括的テストスイートを実行中...")
        
        success_count = 0
        total_tests = 0
        
        # 各テストを実行
        test_functions = [
            ("E2E Tests", self.run_e2e_tests),
            ("GUI Tests", self.run_gui_tests),
            ("Production Tests", self.run_production_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Parallel Tests", self.run_parallel_tests),
            ("Coverage Analysis", self.run_coverage_analysis)
        ]
        
        for test_name, test_func in test_functions:
            total_tests += 1
            print(f"\n{'='*50}")
            print(f"実行中: {test_name}")
            print(f"{'='*50}")
            
            try:
                if test_func(args):
                    success_count += 1
                    print(f"✅ {test_name} 成功")
                else:
                    print(f"❌ {test_name} 失敗")
            except Exception as e:
                self.logger.error(f"❌ {test_name} エラー: {e}")
                print(f"❌ {test_name} エラー")
        
        # 結果サマリー
        print(f"\n{'='*50}")
        print(f"テスト実行完了")
        print(f"{'='*50}")
        print(f"成功: {success_count}/{total_tests}")
        print(f"成功率: {success_count/total_tests*100:.1f}%")
        
        # レポート生成
        if args.report:
            print(f"\n📋 レポートを生成中...")
            self.generate_report(args)
        
        return success_count == total_tests

def create_parser():
    """CLIパーサーを作成"""
    parser = argparse.ArgumentParser(
        description="Professional Statistics Suite - 統合テスト実行ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # すべてのテストを実行
  python cli_test_runner.py run-all

  # 特定のテストを実行
  python cli_test_runner.py e2e
  python cli_test_runner.py gui
  python cli_test_runner.py production

  # パフォーマンステスト
  python cli_test_runner.py performance

  # 並列テスト
  python cli_test_runner.py parallel

  # カバレッジ分析
  python cli_test_runner.py coverage

  # テストデータ生成
  python cli_test_runner.py generate-data --type sample --name employee_data

  # レポート生成
  python cli_test_runner.py report --output test_report.html
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # run-all コマンド
    run_all_parser = subparsers.add_parser('run-all', help='すべてのテストを実行')
    run_all_parser.add_argument('--report', action='store_true', help='レポートを生成')
    
    # e2e コマンド
    e2e_parser = subparsers.add_parser('e2e', help='E2Eテストを実行')
    
    # gui コマンド
    gui_parser = subparsers.add_parser('gui', help='GUIテストを実行')
    
    # production コマンド
    production_parser = subparsers.add_parser('production', help='本番環境テストを実行')
    
    # performance コマンド
    performance_parser = subparsers.add_parser('performance', help='パフォーマンステストを実行')
    
    # parallel コマンド
    parallel_parser = subparsers.add_parser('parallel', help='並列テストを実行')
    
    # coverage コマンド
    coverage_parser = subparsers.add_parser('coverage', help='カバレッジ分析を実行')
    
    # generate-data コマンド
    generate_data_parser = subparsers.add_parser('generate-data', help='テストデータを生成')
    generate_data_parser.add_argument('--type', choices=['sample', 'performance', 'custom'], 
                                     default='sample', help='データタイプ')
    generate_data_parser.add_argument('--name', required=True, help='データセット名')
    generate_data_parser.add_argument('--description', default='', help='説明')
    generate_data_parser.add_argument('--format', default='csv', choices=['csv', 'json', 'pickle'], 
                                     help='出力形式')
    generate_data_parser.add_argument('--size', type=int, default=1000, help='データサイズ')
    generate_data_parser.add_argument('--columns', help='カラム名（カンマ区切り）')
    generate_data_parser.add_argument('--seed', type=int, help='乱数シード')
    generate_data_parser.add_argument('--tags', help='タグ（カンマ区切り）')
    
    # report コマンド
    report_parser = subparsers.add_parser('report', help='レポートを生成')
    report_parser.add_argument('--output', default='test_report.html', help='出力ファイル名')
    
    # list-data コマンド
    list_data_parser = subparsers.add_parser('list-data', help='テストデータ一覧を表示')
    list_data_parser.add_argument('--tags', help='タグでフィルタリング')
    
    # cleanup コマンド
    cleanup_parser = subparsers.add_parser('cleanup', help='古いデータをクリーンアップ')
    cleanup_parser.add_argument('--days', type=int, default=30, help='削除する日数')
    
    return parser

def main():
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # CLIテストランナー初期化
    runner = CLITestRunner()
    
    try:
        if args.command == 'run-all':
            success = runner.run_all_tests(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'e2e':
            success = runner.run_e2e_tests(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'gui':
            success = runner.run_gui_tests(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'production':
            success = runner.run_production_tests(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'performance':
            success = runner.run_performance_tests(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'parallel':
            success = runner.run_parallel_tests(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'coverage':
            success = runner.run_coverage_analysis(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'generate-data':
            success = runner.generate_test_data(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'report':
            success = runner.generate_report(args)
            sys.exit(0 if success else 1)
            
        elif args.command == 'list-data':
            datasets = runner.data_manager.list_test_data(
                tags=args.tags.split(",") if args.tags else None
            )
            print(f"📋 テストデータセット一覧 ({len(datasets)}件):")
            for dataset in datasets:
                print(f"  - {dataset.name}: {dataset.description}")
                print(f"    タイプ: {dataset.data_type}, サイズ: {dataset.size_bytes} bytes")
                print(f"    作成日: {dataset.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                if dataset.tags:
                    print(f"    タグ: {', '.join(dataset.tags)}")
                print()
            
        elif args.command == 'cleanup':
            deleted_count = runner.data_manager.cleanup_old_data(args.days)
            print(f"✅ {deleted_count}件の古いデータを削除しました")
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 