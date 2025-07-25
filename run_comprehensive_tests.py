#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Test Runner
包括的テストランナー

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import sys
import os
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def main():
    """メイン実行関数"""
    print("🚀 Professional Statistics Suite - 包括的テストシステム")
    print("=" * 60)
    
    # 引数解析
    parser = argparse.ArgumentParser(description='包括的テストランナー')
    parser.add_argument('--test-type', choices=['e2e', 'gui', 'production', 'all'], 
                       default='all', help='実行するテストタイプ')
    parser.add_argument('--gui-only', action='store_true', 
                       help='GUIテストのみ実行')
    parser.add_argument('--production-only', action='store_true', 
                       help='本番環境テストのみ実行')
    parser.add_argument('--e2e-only', action='store_true', 
                       help='E2Eテストのみ実行')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='詳細出力')
    parser.add_argument('--no-html-report', action='store_true', 
                       help='HTMLレポート生成を無効化')
    
    args = parser.parse_args()
    
    # テストタイプの決定
    if args.gui_only:
        test_type = 'gui'
    elif args.production_only:
        test_type = 'production'
    elif args.e2e_only:
        test_type = 'e2e'
    else:
        test_type = args.test_type
    
    print(f"📋 実行テストタイプ: {test_type}")
    print(f"🔍 詳細出力: {'有効' if args.verbose else '無効'}")
    print(f"📄 HTMLレポート: {'無効' if args.no_html_report else '有効'}")
    print("-" * 60)
    
    try:
        if test_type == 'all':
            # 統合テストランナーを使用
            from src.tests.integrated_test_runner import IntegratedTestRunner
            
            print("🔄 統合テストランナーを起動中...")
            runner = IntegratedTestRunner()
            
            # HTMLレポート設定
            if args.no_html_report:
                # HTMLレポート生成を無効化する処理を追加
                pass
            
            # 全テスト実行
            results = asyncio.run(runner.run_all_tests())
            
            # 結果表示
            display_results(results, verbose=args.verbose)
            
        else:
            # 個別テスト実行
            run_individual_tests(test_type, verbose=args.verbose)
        
        print("\n✅ テスト実行完了")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_individual_tests(test_type: str, verbose: bool = False):
    """個別テスト実行"""
    print(f"🔄 {test_type.upper()} テストを実行中...")
    
    if test_type == 'e2e':
        from src.tests.e2e_test_automation import E2ETestAutomation
        
        tester = E2ETestAutomation()
        results = asyncio.run(tester.run_comprehensive_e2e_test())
        
    elif test_type == 'gui':
        from src.tests.gui_button_test_automation import GUIButtonTestAutomation
        
        tester = GUIButtonTestAutomation()
        results = tester.run_comprehensive_button_test()
        
    elif test_type == 'production':
        from src.tests.production_environment_test import ProductionEnvironmentTest
        
        tester = ProductionEnvironmentTest()
        results = tester.run_comprehensive_production_test()
    
    # 結果表示
    display_results(results, verbose=verbose)

def display_results(results: dict, verbose: bool = False):
    """結果表示"""
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("=" * 60)
    
    if "error" in results:
        print(f"❌ テスト実行エラー: {results['error']}")
        return
    
    # 実行情報の表示
    if "execution_info" in results:
        exec_info = results["execution_info"]
        print(f"✅ 総テスト数: {exec_info['total_tests']}")
        print(f"✅ 成功テスト数: {exec_info['successful_tests']}")
        print(f"❌ 失敗テスト数: {exec_info['failed_tests']}")
        print(f"📈 成功率: {exec_info['success_rate']:.1f}%")
        print(f"⏱️ 実行時間: {exec_info['duration_seconds']:.2f}秒")
        print(f"🎯 全体ステータス: {results.get('overall_status', 'unknown')}")
        
        # テストシステム詳細
        if "test_systems" in results and verbose:
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
    
    else:
        # 個別テスト結果の表示
        if "total_tests" in results:
            print(f"✅ 総テスト数: {results['total_tests']}")
            print(f"✅ 成功テスト数: {results.get('successful_tests', 0)}")
            print(f"❌ 失敗テスト数: {results.get('failed_tests', 0)}")
            print(f"📈 成功率: {results.get('success_rate', 0):.1f}%")
            print(f"⏱️ 実行時間: {results.get('duration_seconds', 0):.2f}秒")
        
        # 詳細結果の表示
        if verbose and "test_details" in results:
            print("\n📋 詳細結果:")
            for test_name, test_result in results["test_details"].items():
                if "error" in test_result:
                    print(f"  ❌ {test_name}: {test_result['error']}")
                else:
                    print(f"  ✅ {test_name}: 成功")
    
    print("=" * 60)

def check_dependencies():
    """依存関係チェック"""
    print("🔍 依存関係チェック中...")
    
    required_packages = [
        'pandas', 'numpy', 'tkinter', 'psutil', 'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 不足しているパッケージ: {', '.join(missing_packages)}")
        print("以下のコマンドでインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 依存関係チェック完了")
    return True

def check_test_environment():
    """テスト環境チェック"""
    print("🔍 テスト環境チェック中...")
    
    # ディレクトリ存在チェック
    required_dirs = ['src', 'logs', 'test_results']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"📁 ディレクトリ作成: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
    
    # テストファイル存在チェック
    test_files = [
        'src/tests/e2e_test_automation.py',
        'src/tests/gui_button_test_automation.py',
        'src/tests/production_environment_test.py',
        'src/tests/integrated_test_runner.py'
    ]
    
    missing_files = []
    for file_path in test_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 不足しているテストファイル:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ テスト環境チェック完了")
    return True

if __name__ == "__main__":
    # 環境チェック
    if not check_dependencies():
        sys.exit(1)
    
    if not check_test_environment():
        sys.exit(1)
    
    # メイン実行
    main() 